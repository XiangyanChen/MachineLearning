import numpy as np

def calcQ(thetaA, thetaB, coin, pz1, pz2):
    numerator = pz1 * (thetaA ** int(coin[0])) * ((1 - thetaA) ** int(coin[1]))  # 分子
    denominator = numerator + pz1 * (thetaB ** int(coin[0])) * ((1 - thetaB) ** int(coin[1]))  # 分母
    return numerator / denominator  #返回Q值

if __name__ == '__main__':
    coins = [[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]]    #五个硬币次数
    N = len(coins)
    thetaA = 0.6    #初始值thetaA
    thetaB = 0.5    #初始值thetaB
    pz1 = 0.5   #硬币的概率都是一样的0.5
    pz2 = 0.5
    maxIter = 10
    for n in range(maxIter):    #题目要求，循环10次
        coinsAResult = [0, 0]
        coinsBResult = [0, 0]
        for i in range(N):  #循环N次计算出Q值
            Q_a = calcQ(thetaA, thetaB, coins[i], pz1, pz2) #E-step,计算QA
            Q_b = calcQ(thetaB, thetaA, coins[i], pz1, pz2)
            coinsAResult[0] += Q_a * coins[i][0]    #M-step，为计算theta而准备，对应第一步的21.3H
            coinsAResult[1] += Q_a * coins[i][1]    #对应第一步的8.6T
            coinsBResult[0] += Q_b * coins[i][0]    #对应第一步的11.7H
            coinsBResult[1] += Q_b * coins[i][1]    #对应第一步的8.4T
        thetaA = coinsAResult[0] / (coinsAResult[1] + coinsAResult[0])  #计算theta
        thetaB = coinsBResult[0] / (coinsBResult[1] + coinsBResult[0])
        print("第%d次，thetaA:%.2f" % (n+1, thetaA))  #输出每次迭代的thetaA
        print("第%d次，thetaB:%.2f" % (n+1, thetaB))  #输出每次迭代的thetaB

