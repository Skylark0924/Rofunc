function QM=QuatMatrixConj(q)

QM =      [q.s    -q.v(1) -q.v(2) -q.v(3);
           q.v(1) q.s     q.v(3) -q.v(2);
           q.v(2) -q.v(3)  q.s     q.v(1);
           q.v(3) q.v(2) -q.v(1)  q.s   ];


end