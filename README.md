# nolearn-lasagne-theano

## Description
Sample for setting up a conda virtualenv for running nolearn+lasagne+theano on Ubuntu

## Steps

1. Install CUDA 7.5
2. Install anconda 
 * ```wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.5.0-Linux-x86_64.sh```
 * ```bash Anaconda2-2.5.0-Linux-x86_64.sh```
3. Install git ```sudo apt-get git git-core```
4. Get this repo ```git clone https://github.com/EliSchwartz/nolearn-lasagne-theano.git```
5. Run sample:
 * ```cd nolearn-lasagne-theano```
 * ```source startup.sh```
 * ```python sample.py```

The output should look like:


Using gpu device 0: GeForce GT 630 (CNMeM is disabled)
/# Neural Network with 22 learnable parameters

/## Layer information

  #    name    size
/---  ------  ------
  0              10
                  2

  epoch    train loss    valid loss    train/val  dur
/-------  ------------  ------------  -----------  -----
      1       1.75000       0.73575      2.37852  0.00s
      2       0.43073       0.04950      8.70145  0.00s
      3       0.06907       0.17868      0.38656  0.00s
      4       0.22627       0.32091      0.70507  0.00s
      5       0.27501       0.22801      1.20613  0.00s
      6       0.16293       0.08731      1.86621  0.00s
      7       0.06189       0.03197      1.93584  0.00s
      8       0.03020       0.03328      0.90749  0.00s
      9       0.02899       0.03325      0.87193  0.00s
     10       0.02489       0.02075      1.19975  0.00s
Score on test:0.0174257978797

