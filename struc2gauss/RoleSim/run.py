import os

ids = [300]
for item in ids:
	for i in range(5):
		cmd = './main.o 200 /home/ypei1/Embedding/Syn/Repeat/sbm.' + str(item) + '.' + str(i + 6) + '.edgelist /home/ypei1/Embedding/Syn/Repeat/sbm.' + str(item) + '.' + str(i + 6) + '.sim'
		print cmd
		os.system(cmd)