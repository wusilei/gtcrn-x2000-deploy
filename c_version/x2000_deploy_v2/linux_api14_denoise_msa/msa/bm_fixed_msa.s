	.file	1 "bm_fixed_msa.c"
	.section .mdebug.abi32
	.previous
	.nan	2008
	.abicalls
	.option	pic0
	.text
	.align	2
	.globl	bm_fixed_msa
	.set	nomips16
	.set	nomicromips
	.ent	bm_fixed_msa
	.type	bm_fixed_msa, @function
bm_fixed_msa:
	.frame	$sp,40,$31		# vars= 0, regs= 9/0, args= 0, gp= 0
	.mask	0x40ff0000,-4
	.fmask	0x00000000,0
	lui	$2,%hi(done.2107)
	addiu	$sp,$sp,-40
	lw	$3,%lo(done.2107)($2)
	sw	$16,4($sp)
	move	$16,$5
	sw	$fp,36($sp)
	sw	$23,32($sp)
	sw	$22,28($sp)
	sw	$21,24($sp)
	sw	$20,20($sp)
	sw	$19,16($sp)
	sw	$18,12($sp)
	.set	noreorder
	.set	nomacro
	bne	$3,$0,$L2
	sw	$17,8($sp)
	.set	macro
	.set	reorder

	li	$3,1			# 0x1
	sw	$3,%lo(done.2107)($2)
#APP
 # 24 "msa/bm_fixed_msa.c" 1
	mfc0 $2, $12
	or $2, $2, 0x20000000
	mtc0 $2, $12
 # 0 "" 2
 # 25 "msa/bm_fixed_msa.c" 1
	.set push; .set mips32r2; mfc0 $2, $16, 5
	or $2, $2, 0x400
	mtc0 $2, $16, 5; .set pop
 # 0 "" 2
 # 26 "msa/bm_fixed_msa.c" 1
	.set push; .set mips32r2; mfc0 $2, $16, 5
	or $2, $2, 0x08000000
	mtc0 $2, $16, 5; .set pop
 # 0 "" 2
#NO_APP
$L2:
	li	$2,65535			# 0xffff
	addiu	$7,$4,1028
	fill.w	$w4,$2
	addiu	$15,$6,500
	addiu	$14,$4,260
	addiu	$fp,$4,3344
	move	$25,$6
	move	$24,$4
	addiu	$23,$16,120
	li	$22,16			# 0x10
	li	$21,2			# 0x2
	li	$20,64			# 0x40
	li	$19,65			# 0x41
	li	$18,1			# 0x1
	li	$12,3			# 0x3
$L14:
	subu	$2,$24,$4
	addiu	$5,$14,-244
	sltu	$5,$25,$5
	.set	noreorder
	.set	nomacro
	beq	$5,$0,$L20
	subu	$3,$25,$6
	.set	macro
	.set	reorder

	addiu	$5,$15,-484
	sltu	$5,$24,$5
	bne	$5,$0,$L17
$L20:
	srl	$10,$24,2
	subu	$10,$0,$10
	andi	$10,$10,0x3
	.set	noreorder
	.set	nomacro
	beq	$10,$0,$L5
	move	$5,$0
	.set	macro
	.set	reorder

	lw	$5,0($24)
	.set	noreorder
	.set	nomacro
	beq	$10,$18,$L18
	sw	$5,0($25)
	.set	macro
	.set	reorder

	lw	$5,4($24)
	.set	noreorder
	.set	nomacro
	bne	$10,$12,$L19
	sw	$5,4($25)
	.set	macro
	.set	reorder

	lw	$9,8($24)
	li	$8,62			# 0x3e
	li	$5,3			# 0x3
	sw	$9,8($25)
$L6:
	subu	$9,$19,$10
	sll	$10,$10,2
	addu	$2,$10,$2
	addu	$3,$10,$3
	addu	$2,$4,$2
	addu	$3,$6,$3
	ld.w	$w0,0($2)
	srl	$10,$9,2
	addiu	$2,$2,16
	addiu	$3,$3,16
	st.w	$w0,-16($3)
$L7:
	ld.w	$w0,0($2)
	st.w	$w0,0($3)
	ld.w	$w0,16($2)
	st.w	$w0,16($3)
	ld.w	$w0,32($2)
	st.w	$w0,32($3)
	ld.w	$w0,48($2)
	st.w	$w0,48($3)
	ld.w	$w0,64($2)
	st.w	$w0,64($3)
	ld.w	$w0,80($2)
	st.w	$w0,80($3)
	ld.w	$w0,96($2)
	st.w	$w0,96($3)
	ld.w	$w0,112($2)
	st.w	$w0,112($3)
	ld.w	$w0,128($2)
	st.w	$w0,128($3)
	ld.w	$w0,144($2)
	st.w	$w0,144($3)
	ld.w	$w0,160($2)
	st.w	$w0,160($3)
	ld.w	$w0,176($2)
	st.w	$w0,176($3)
	ld.w	$w0,192($2)
	st.w	$w0,192($3)
	ld.w	$w0,208($2)
	.set	noreorder
	.set	nomacro
	bne	$10,$22,$L33
	st.w	$w0,208($3)
	.set	macro
	.set	reorder

	ld.w	$w0,224($2)
	addiu	$5,$5,64
	addiu	$8,$8,-64
	.set	noreorder
	.set	nomacro
	beq	$9,$20,$L9
	st.w	$w0,224($3)
	.set	macro
	.set	reorder

$L16:
	sll	$2,$5,2
	addiu	$3,$8,-1
	addu	$9,$24,$2
	addu	$5,$25,$2
	lw	$9,0($9)
	.set	noreorder
	.set	nomacro
	beq	$3,$0,$L9
	sw	$9,0($5)
	.set	macro
	.set	reorder

	addiu	$3,$2,4
	addu	$5,$24,$3
	addu	$3,$25,$3
	lw	$5,0($5)
	.set	noreorder
	.set	nomacro
	beq	$8,$21,$L9
	sw	$5,0($3)
	.set	macro
	.set	reorder

	addiu	$2,$2,8
	addu	$3,$24,$2
	addu	$2,$25,$2
	lw	$3,0($3)
	sw	$3,0($2)
$L9:
	addiu	$13,$15,-240
$L34:
	move	$8,$16
$L11:
	ldi.w	$w2,0
	move	$5,$8
	move	$2,$14
$L10:
	ld.w	$w0,0($5)
	lw	$3,0($2)
	addiu	$2,$2,4
	addiu	$5,$5,128
	fill.w	$w1,$3
	srli.w	$w3,$w0,16
	and.v	$w0,$w0,$w4
	ilvr.w	$w0,$w3,$w0
	slli.w	$w0,$w0,15
	.set	noreorder
	.set	nomacro
	bne	$2,$7,$L10
	madd_q.w	$w2,$w1,$w0
	.set	macro
	.set	reorder

	copy_s.w	$10,$w2[0]
	addiu	$13,$13,16
	copy_s.w	$9,$w2[1]
	copy_s.w	$5,$w2[2]
	copy_s.w	$3,$w2[3]
	sll	$10,$10,1
	sll	$9,$9,1
	sw	$10,-16($13)
	sll	$5,$5,1
	sw	$9,-12($13)
	sll	$3,$3,1
	sw	$5,-8($13)
	sw	$3,-4($13)
	.set	noreorder
	.set	nomacro
	bne	$13,$15,$L11
	addiu	$8,$8,8
	.set	macro
	.set	reorder

	move	$5,$23
	addiu	$17,$13,16
$L13:
	move	$8,$5
	move	$3,$14
	move	$10,$0
	move	$11,$0
$L12:
	mtlo	$10
	addiu	$3,$3,4
	lhu	$9,0($8)
	addiu	$8,$8,128
	lw	$10,-4($3)
	mthi	$11
	madd	$10,$9
	mflo	$10
	.set	noreorder
	.set	nomacro
	bne	$3,$2,$L12
	mfhi	$11
	.set	macro
	.set	reorder

	addiu	$3,$10,16384
	addiu	$13,$13,4
	sltu	$8,$3,$10
	srl	$3,$3,15
	addu	$8,$8,$11
	addiu	$5,$5,2
	sll	$8,$8,17
	or	$3,$8,$3
	.set	noreorder
	.set	nomacro
	bne	$17,$13,$L13
	sw	$3,-4($13)
	.set	macro
	.set	reorder

	addiu	$14,$14,1028
	addiu	$24,$24,1028
	addiu	$25,$25,516
	addiu	$7,$7,1028
	.set	noreorder
	.set	nomacro
	bne	$14,$fp,$L14
	addiu	$15,$15,516
	.set	macro
	.set	reorder

	lw	$fp,36($sp)
	lw	$23,32($sp)
	lw	$22,28($sp)
	lw	$21,24($sp)
	lw	$20,20($sp)
	lw	$19,16($sp)
	lw	$18,12($sp)
	lw	$17,8($sp)
	lw	$16,4($sp)
	.set	noreorder
	.set	nomacro
	jr	$31
	addiu	$sp,$sp,40
	.set	macro
	.set	reorder

$L18:
	li	$8,64			# 0x40
	.set	noreorder
	.set	nomacro
	b	$L6
	li	$5,1			# 0x1
	.set	macro
	.set	reorder

$L5:
	addu	$2,$4,$2
	addu	$3,$6,$3
	ld.w	$w0,0($2)
	addiu	$3,$3,16
	addiu	$2,$2,16
	li	$8,65			# 0x41
	li	$9,65			# 0x41
	li	$10,16			# 0x10
	.set	noreorder
	.set	nomacro
	b	$L7
	st.w	$w0,-16($3)
	.set	macro
	.set	reorder

$L19:
	li	$8,63			# 0x3f
	.set	noreorder
	.set	nomacro
	b	$L6
	li	$5,2			# 0x2
	.set	macro
	.set	reorder

$L17:
	move	$3,$25
	move	$2,$24
$L3:
	lw	$5,0($2)
	addiu	$3,$3,4
	addiu	$2,$2,4
	.set	noreorder
	.set	nomacro
	bne	$2,$14,$L3
	sw	$5,-4($3)
	.set	macro
	.set	reorder

	.set	noreorder
	.set	nomacro
	b	$L34
	addiu	$13,$15,-240
	.set	macro
	.set	reorder

$L33:
	addiu	$5,$5,60
	.set	noreorder
	.set	nomacro
	b	$L16
	addiu	$8,$8,-60
	.set	macro
	.set	reorder

	.end	bm_fixed_msa
	.size	bm_fixed_msa, .-bm_fixed_msa
	.local	done.2107
	.comm	done.2107,4,4
	.ident	"GCC: (Ingenic Linux-Release5.1.8-Default_xburst2_glibc2.29 Optimize: jr base on 5.1.6 2023.07-18 08:46:58) 7.2.0"
