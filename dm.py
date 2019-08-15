import pandas
import numpy
import sklearn
from sklearn.naive_bayes import GaussianNB
#Binary Classification 
#Gaussian Naive Bayes Classifier
#REmoved of Name, Cabin, Ticket

df = pandas.read_csv('train.csv', header = 0, usecols = [0, 2, 4, 5, 6,7,8, 9,10, 11])


#print(df.dtypes)
#print(df.count )
gb = GaussianNB()

n = numpy.array(df)


for x in n:
	if x[2] == 'male':
		x[2] = 0
	else:
		x[2] = 1

for x in n:
	if x[9] == 'S':
		x[9] = 1
	elif x[9] == 'C':
		x[9] = 2
	elif x[9] =='Q':
		x[9] = 3
	else:
		x[9] = 0	
for x in n:
	x[6] = 0
	x[8] = 0
for x in n:
	if x[7] <= 10:
		x[7] = 10
	elif x[7] <= 20:
		x[7] = 20
	elif x[7] <= 30:
		x[7] = 30
	elif x[7] <= 40:
		x[7] = 40
	elif x[7] <= 50:
		x[7] = 50
	elif x[7] <= 60:
		x[7] = 60
	elif x[7] <= 70:
		x[7] = 70
	elif x[7] <= 80:
		x[7] = 80
	elif x[7] <= 90:
		x[7] = 90
	elif x[7] <= 100:
		x[7] = 100
	elif x[7] <= 110:
		x[7] = 110
	elif x[7] <= 110:
		x[7] = 110
	elif x[7] <= 120:
		x[7] = 120
	elif x[7] <= 130:
		x[7] = 130
	elif x[7] <= 140:
		x[7] = 140
	elif x[7] <= 150:
		x[7] = 150 
	elif x[7] <= 160:
		x[7] = 160
	elif x[7] <= 170:
		x[7] = 170
	elif x[7] >= 500:
		x[7] = 500
	elif x[7] >= 400:
		x[7] = 400
	elif x[7] >= 300:
		x[7] = 300
	else:
		x[7] = 200

for x in n:
	if numpy.isnan(x[3]):
		x[3] = 0
	elif x[3] <= 10:
		x[3] = 5
	elif x[3] <= 20:
		x[3] = 10
	elif x[3] <= 30:
		x[3] = 20
	elif x[3] <= 40:
		x[3] = 30
	elif x[3] <= 50:
		x[3] = 40
	elif x[3] <= 60:
		x[3] = 50
	else:
		x[3] = 60
	








df = pandas.read_csv('train.csv', header = 0, usecols = [0,1])

t = numpy.array(df['Survived'])
model = gb.fit(n, t)

df = pandas.read_csv('test.csv', header = 0, usecols = [0,1,3,4,5,6,7,8,9,10])
n = numpy.array(df)

#t = numpy.array(d['Survived'])
for x in n:
	if x[2] == 'male':
		x[2] = 0
	else:
		x[2] = 1
for x in n:
	if numpy.isnan(x[3]):
		x[3] = 0

for x in n:
	if x[9] == 'S':
		x[9] = 1
	elif x[9] == 'C':
		x[9] = 2
	elif x[9] == 'Q':
		x[9] = 3
	else:
		x[9] = 0
for x in n:
	if numpy.isnan(x[7]):
		x[7] = 0
	x[6] = 0
	x[7] = 0
	x[8] = 0


f = open("output.txt", "w")
answer = gb.predict(n)
count = 0
df = pandas.read_csv('test.csv', header = 0, usecols = [0])
d = numpy.array(df)
print("PassengerId,Survived", file = f)
for x in answer:
	print(str(d[count][0]) + "," + str(x), file = f)
	count += 1
f.close()
#How to handle Nan information