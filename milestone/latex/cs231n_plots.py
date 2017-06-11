import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

baseline=[0.2858082191780822,0.4055342465753425,0.47715068493150686,0.5312602739726028,0.5726027397260274]
resnet18=[.3819178082191781,0.514317491748194,0.5919481845571832,0.646837571838181,0.6873190129101192]
resnet18_proj=[0.36747945205479454,0.5028767123287671,0.5813150684931507,0.6330958904109589,0.6727671232876712]
resnet34=[0.38652054794520546,0.5182739726027398,0.5958082191780822,0.6491232876712328,0.6886301369863014]
resnet50=[0.37583561643835617,0.5126849315068494,.591013698630137,0.6428219178082192,0.682958904109589]

x=[1,2,3,4,5]

plt.figure(figsize=(20,20))
baseline,=plt.plot(x,baseline,linewidth=3)
resnet18,=plt.plot(x,resnet18,linewidth=3)
resnet18_proj,=plt.plot(x,resnet18_proj,linewidth=3)
resnet34,=plt.plot(x,resnet34,linewidth=3)
resnet50,=plt.plot(x,resnet50,linewidth=3)
plt.axis([1,5,0,1.0])

plt.xlabel('Top-n',size=25)
plt.ylabel('Accuracy',size=25)

plt.xticks(np.linspace(1, 5, num=5, endpoint=True, retstep=False, dtype=None),fontsize = 20)
plt.yticks(np.linspace(0, 1.0, num=21, endpoint=True, retstep=False, dtype=None),fontsize = 20)


plt.legend([baseline,resnet18,resnet18_proj,resnet34,resnet50],['Baseline','ResNet 18','ResNet 18 Proj','ResNet 34','ResNet 50'],prop={'size':25})
plt.title("Accuracy Comparison for top-n accuracy when varying n",size=25)
#plt.show()
plt.savefig('top-n.png')
