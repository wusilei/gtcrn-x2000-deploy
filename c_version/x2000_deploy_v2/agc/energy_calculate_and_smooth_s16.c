/******************************************************************************
Description     :   求帧数据的平均能量(绝对值求和后取平均)，可根据需要进行帧间平滑
                    energy_out = history_fac*energy_out + (0x7fff-history_fac)*current_energy
                    如果不想做帧间平滑，直接将history_fac置0即可
                 
Input           : 	data_in，    输入数据
                    data_len，   数据长度,最大2^15
                    history_fac，记忆因子,(0~1)*2^15
                                 
InOut           :   

Output          :   *energy_out，平滑后的能量输出
                    			
Return          :   current_energy，本帧的能量(未平滑)
Date            :   2020.05.27
Author          :   LongXueKun
*******************************************************************************/
unsigned short energy_calculate_and_smooth_s16( short *data_in,
                                                unsigned short data_len,
                                                unsigned short history_fac,
                                                unsigned short *energy_out)
{
    short data_temp1;
    unsigned short data_temp2;
    unsigned short current_energy;
    unsigned int data_sum = 0;
    unsigned int i;

    for(i=0; i<data_len; i++)
    {
        data_temp1 = data_in[i];
        data_temp2 = (data_temp1<0)? (-data_temp1) : data_temp1;
        data_sum += data_temp2;
    }
    current_energy = (unsigned short)(data_sum/data_len);

    *energy_out = (unsigned short)((((unsigned int)history_fac*(*energy_out)) + (((unsigned int)0x7fff-history_fac)*current_energy)+16384) >> 15);
    
    return current_energy;
}



