; A000045: Fibonacci numbers.
mov $3,1
lpb $0
  sub $0,1
  mov $2,$1
  add $1,$3
  mov $3,$2
lpe
mov $0,$1
