
BEGIN {
  for (i = 1; i <= 6; i++) {
    c1[i] = 0;
    c2[i] = 0;
    c3[i] = 0;
    c4[i] = 0;
    c5[i] = 0;
    c6[i] = 0;
  }
}

{
  split($(NF-1), array, ")")
  if (array[1] == 0) c1[$2+1]++;
  else if (array[1] == 1) c2[$2+1]++;
  else if (array[1] == 2) c3[$2+1]++;
  else if (array[1] == 3) c4[$2+1]++;
  else if (array[1] == 4) c5[$2+1]++;
  else if (array[1] == 5) c6[$2+1]++;
}

END {
  for (i=1;i<=6;i++) printf("%d ", c1[i]);
  print ""
  for (i=1;i<=6;i++) printf("%d ", c2[i]);
  print ""
  for (i=1;i<=6;i++) printf("%d ", c3[i]);
  print ""
  for (i=1;i<=6;i++) printf("%d ", c4[i]);
  print ""
  for (i=1;i<=6;i++) printf("%d ", c5[i]);
  print ""
  for (i=1;i<=6;i++) printf("%d ", c6[i]);
  print ""
}

