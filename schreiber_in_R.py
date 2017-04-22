'''
s=1
X=c(1,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1)
X
Y=c(1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1)
Y
L4=L1=length(X)-s # Lengths of vector Xn+1.
L3=L2=length(X) # Lengths of vector Xn (and Yn).

TPvector1=rep(0,L1) # Init.
TPvector1

for(i in 1:L1)
{
        TPvector1[i]=paste(c(X[i+s],"i",X[i],"i",Y[i]),collapse="") # "addresses"
}
TPvector1
table(TPvector1)

TPvector1T=table(TPvector1)/length(TPvector1) # Table of probabilities.
TPvector1T

TPvector2=X
table(X)
TPvector2T=table(X)/sum(table(X))
TPvector2T


TPvector3=rep(0,L3)

for(i in 1:L3)
{
        TPvector3[i]=paste(c(X[i],"i",Y[i]),collapse="") # addresses
}
TPvector3

TPvector3T=table(TPvector3)/length(TPvector2)
TPvector3T

    #----------------#
    # 4. p(Xn+s,Xn): #
    #----------------#

TPvector4=rep(0,L4)

for(i in 1:L4)
{
        TPvector4[i]=paste(c(X[i+s],"i",X[i]),collapse="") # addresses
}
TPvector4

TPvector4T=table(TPvector4)/length(TPvector4)
TPvector4T

    #--------------------------#
    # Transfer entropy T(Y->X) #
    #--------------------------#

SUMvector=rep(0,length(TPvector1T))
L1
for(n in 1:length(TPvector1T))
{
        SUMvector[n]=TPvector1T[n]*log10((TPvector1T[n]*TPvector2T[(unlist(strsplit(names(TPvector1T)[n],"i")))[2]])/(TPvector3T[paste((unlist(strsplit(names(TPvector1T)[n],"i")))[2],"i",(unlist(strsplit(names(TPvector1T)[n],"i")))[3],sep="",collapse="")]*TPvector4T[paste((unlist(strsplit(names(TPvector1T)[n],"i")))[1],"i",(unlist(strsplit(names(TPvector1T)[n],"i")))[2],sep="",collapse="")])) #afsddgfd
}
SUMvector
sum(SUMvector)

0.01944000  0.03654962 -0.02229364  0.01784839 -0.01384429 -0.01677060 0.02307381
0.023073, 0.127429, -0.023408, 0.008284, 0.019439, -0.01677059, 0.023146
'''