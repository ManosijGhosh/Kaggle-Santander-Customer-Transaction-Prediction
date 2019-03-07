fileRead = open('sample_submission.csv','r') 
file = open('sample_submission_2.csv','w') 
file.write('ID_code,target\n') 
for i in range(0,200000):
	string = fileRead.readline()
	string = string[0:len(string)-3]+'\n'
	file.write(string)