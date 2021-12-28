import sys
nwater = int(sys.argv[1])
jname = f'water{nwater}_gpu.out'
with open(jname, 'r') as f:
    print(jname)
    header = f.readline()
    for rc in range(2, 12, 2):
        for kappa in range(10):
            print_flag = True
            kappa = (kappa+1)/10
            previous_etotal = 0
            for kmax in range(5, 300, 5):
                line = f.readline()
                rc, kappa, kmax, time, *_, etotal = list(map(float, line.split()))
                delta_etotal = abs(abs(previous_etotal - etotal)/etotal)
                if kappa == 0.3 and kmax==10:
                    print(f'{time=}')
                if delta_etotal < 1e-4 and print_flag:
                    print(rc, kappa, kmax)
                    print_flag = False
                else:
                    # print(delta_etotal)
                    previous_etotal = etotal
