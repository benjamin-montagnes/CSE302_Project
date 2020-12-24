.L0:
  %0.0 = const 42;
  %1.1 = copy %0.0;
  %2.2 = const 84;
  %3.3 = copy %2.2;
  %4.4 = add %1.1, %3.3;
  param 1, %4.4;
  %_ = call @__bx_print_int, 1;
  %5.6 = add %1.1, %3.3;
  %6.7 = copy %5.6;
  param 1, %6.7;
  %_ = call @__bx_print_int, 1;
  jmp .L1;
.L1:
  ret %_;

