import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes




# DQN = [165,193,156,211,57,233,301,281,198,377,422,424,357,415,366,356,431,425,437,429,413]
# double_DQN =  [16,212,143,144,201,219,220,269,237,351,325,365,371,349,356,429,394,435,388,445,466]
# double_Q_learning = [225,138,262,37,231,231,152,315,261,349,424,461,437,430,472,502,459,527,494,524,501]


resnet18_train = [0.3067,0.3770684931506849,0.423013698630137,0.46216438356164385,0.49,0.528958904109589,0.5601917808219178,0.5963287671232876]
resnet18_val = [0.28416438356164386,0.33936986301369865,0.3648767123287671,0.3774246575342466,0.3823561643835616,0.3819178082191781,0.3803013698630137,0.3734246575342466]

resnet32_train = [0.2913972602739726,0.36994520547945203,0.4084109589041096,0.4430958904109589,0.488,0.5191780821917809,0.5595616438356165,0.5965753424657534]
resnet32_val = [0.27,0.3264931506849315,0.3515342465753425,0.37046575342465754,0.38115068493150683,0.37967123287671234,0.3779178082191718,0.37465753424657533]

baseline = []
Epoch_resnet18 = [1,2,3,4,5,6,7,8]
Epoch_resnet32 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
Epoch_baseline = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

plt.figure(figsize=(20,20))

DQN,=plt.plot(Epoch,DQN,linewidth=3)
double_DQN,=plt.plot(Epoch,double_DQN,linewidth=3)
double_Q_learning,=plt.plot(Epoch,double_Q_learning,linewidth=3)

plt.axis([1,7,0,100])

plt.xlabel('Epoch',size=25)
plt.ylabel('Average Reward',size=25)

plt.xticks(np.linspace(0, 20, num=11, endpoint=True, retstep=False, dtype=None),fontsize = 20)
plt.yticks(np.linspace(0, 550, num=21, endpoint=True, retstep=False, dtype=None),fontsize = 20)


plt.legend([DQN,double_DQN,double_Q_learning],['DQN','double DQN','double Q-Learning'],prop={'size':25})
plt.title("Performance Comparison of DQN, Double DQN and Double Q-Learning  ",size=25)
# plt.show()
plt.savefig('milestone.png')