%% export_intermediate.m — 导出单帧所有中间层数据用于 Python 对比
%% 用法: 在 MATLAB 中 run export_intermediate.m
%% 输出: intermediate_f0.mat 包含所有层的 s32f20 整数值

close all; clear; clc;
addpath 'para_in_mat'; addpath 'test_wavs';

%% Load Para
erbfc_weight = ( importdata('erb_erb_fc_weight.mat') ).';
ierbfc_weight = ( importdata('erb_ierb_fc_weight.mat') ).';

%% Temporal cache (initialize to zeros)
conv_prev_enc = zeros(16,16,33);
h_prev_enc = zeros(3,16);
inter_prev1 = zeros(33,16);
inter_prev2 = zeros(33,16);
conv_prev_dec = zeros(16,16,33);
h_prev_dec = zeros(3,16);

%% Load audio (first frame only)
[noisy_audio, fs] = audioread('noisy_fileid_5.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('win_hann.mat');

[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);
T = 1;  % Only process first frame

%% Process frame 0
t = 1;
fprintf('Processing frame %d...\n', t);

x = mag_gen(cmp_real(t,:), cmp_imag(t,:));

%% BM
x_bm = BM_module(x, erbfc_weight);

%% SFE
x_sfe = SFE_module(x_bm);

%% Encoder
[y_conv0, y_conv1, y_conv2, y_conv3, y_conv4, conv_prev_enc, h_prev_enc] = ...
    Encoder_module(x_sfe, conv_prev_enc, h_prev_enc);

% Export encoder intermediates as raw integers (s32f20)
data = struct();
data.x          = x;          % mag_gen output [3,257]
data.x_bm       = x_bm;       % BM output [3,129]
data.x_sfe      = x_sfe;      % SFE output [9,129]
data.y_conv0    = y_conv0;    % Conv0 output [16,65]
data.y_conv1    = y_conv1;    % Conv1 output [16,33]
data.y_conv2    = y_conv2;    % GT-Conv0 output [16,33]
data.y_conv3    = y_conv3;    % GT-Conv1 output [16,33]
data.y_conv4    = y_conv4;    % GT-Conv2 output [16,33]

% Also export GT-Conv internal layers (call them individually with zero state)
% GT-Conv0 internals
x1 = y_conv1(1:8,:);
x1_sfe = SFE_module(x1);
% P-Conv-0
x1_pconv0 = P_Conv_block_0(x1_sfe, 0);
% DD-Conv
[x1_ddconv, ~] = DD_Conv_block(x1_pconv0, zeros(16,2,33), 1, 0);
% P-Conv-1
x1_pconv1 = P_Conv_block_1(x1_ddconv, 0);
% TRA (with zero state)
[x1_tra, ~] = TRA_module(x1_pconv1, zeros(1,16), 0);

data.gtc0_x1_sfe    = x1_sfe;
data.gtc0_x1_pconv0 = x1_pconv0;
data.gtc0_x1_ddconv = x1_ddconv;
data.gtc0_x1_pconv1 = x1_pconv1;
data.gtc0_x1_tra    = x1_tra;

%% GDPRNN
[y_rnn1, inter_prev1] = GDPRNN_module(y_conv4, inter_prev1, 1);
[y_rnn2, inter_prev2] = GDPRNN_module(y_rnn1, inter_prev2, 2);

data.y_rnn1 = y_rnn1;
data.y_rnn2 = y_rnn2;

%% Decoder
[y_dec, conv_prev_dec, h_prev_dec] = Decoder_module(y_rnn2, y_conv0, y_conv1, y_conv2, y_conv3, y_conv4, ...
    conv_prev_dec, h_prev_dec);

data.y_dec = y_dec;

% GT-DeConv0 internals
x_dec_in = y_rnn2;
x_dec_skip = x_dec_in + y_conv4;
x1_dec = x_dec_skip(1:8,:);
x1_dec_sfe = SFE_module(x1_dec);
x1_depconv0 = P_DeConv_block_0(x1_dec_sfe, 0);
[x1_deddconv, ~] = DD_DeConv_block(x1_depconv0, zeros(16,8,33), 5, 0);
x1_depconv1 = P_DeConv_block_1(x1_deddconv, 0);
[x1_detra, ~] = DeTRA_module(x1_depconv1, zeros(1,16), 0);

data.gtd0_x1_sfe      = x1_dec_sfe;
data.gtd0_x1_depconv0 = x1_depconv0;
data.gtd0_x1_deddconv = x1_deddconv;
data.gtd0_x1_depconv1 = x1_depconv1;
data.gtd0_x1_detra    = x1_detra;

%% BS + MASK
y_bs = BS_module(y_dec, ierbfc_weight);
y_mask = MASK_module(y_bs, x(2,:), x(3,:));

data.y_bs   = y_bs;
data.y_mask = y_mask;
data.crm    = y_mask * 2^(-20);  % dequantized CRM

%% Save
save('intermediate_f0.mat', '-struct', 'data');
fprintf('Saved intermediate_f0.mat with %d fields\n', numel(fieldnames(data)));

%% Print summary
fns = fieldnames(data);
for i = 1:numel(fns)
    d = data.(fns{i});
    fprintf('  %-25s size=[%s]  range=[%.2f, %.2f]\n', ...
        fns{i}, strjoin(cellstr(num2str(size(d))), 'x'), min(d(:)), max(d(:)));
end
