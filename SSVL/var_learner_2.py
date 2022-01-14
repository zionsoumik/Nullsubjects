'''
code for variational learner modeling the null subject acquisition in English. The normal
variational learner works and thus I have the entire thing about null subjects coded. uncomment
ch=2 line for vanilla variational learner.
'''

from __future__ import division
import csv
import matplotlib
matplotlib.use('Agg')
import math
from multiprocessing import Queue
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
colag_file=open("COLAG_2011_flat_formatted.txt")
# readers are csv readers which is a library thing nothing impactful
results=[]
LD={} # LD is for the regular language domain
### choosegrammar looks at the probabities for all the parameters and  chooses the current grammar
def choosegrammar(grammar):
    ret=""
    for g in grammar:
        #gl=random.choice([0,1],p=[1-g,g])
        rand=random.random()
        if rand < g:
            ret = ret + str(1)
        else:
            ret = ret + str(0)
        #ret=ret+str(gl)
    return ret

# this part sets the dictionary keys which are the grammars
def var_learner(r,R,IMP_list,DEC_Q_list,max_sentences,Gtarg,q,sl,c,growth):
    global LD
    global results
    ####
    weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #r=0.001
    faculty=0 # faculty is a variable indicating the ability of child to distinguish
    ns_list=[]
    for i in range(0,max_sentences): # iterating till max_sentences
        if growth==1:
		faculty=sl*i+c	
		if faculty < 0:
              		faculty = 0
        if faculty > 1:
              		faculty = 1
        else:
    	    	faculty=1/(1+math.exp(-c*(i-sl)))
            #faculty=1/(1+math.exp(-0.000000993*(i-5516232.93)))
        #faculty=0
    # the above block checks if faculty does not exceed 1 or fall below 0. if you look closely faculty
    # is 0 for sentence no. below max_sentences*0.45 and goes on to linearly increase for another
    # max_sentences*0.4 sentences.

        ch = random.choice([1, 2], p=[1 - faculty, faculty])
        # ch is choice if faulty is x then choice 1 will be chosen (1-x)% of the time. else it chooses
        # the other choice 2. choice 1 is basically IMP replaced by DEC. choice 2 is fully formed faculty
        # i.e. adult IMP and DEC differentiation
        ch=2 #uncomment this to get the pure var learner which mmeans it will choose choice 2 all time
        grammar=""
        '''
            l is the current grammar. we choose l untill the l in in colag 
            domain and choosegrammar() has the probabilities right.
        '''
        while grammar not in LD.keys():
            grammar = choosegrammar(weights)
	sent_choice=0
	if i<3566210:
            sent_choice = random.choice([1, 2], p=[0.16, 0.84])
	else:
	    sent_choice = random.choice([1, 2], p=[0.18, 0.82])

        if sent_choice == 1:
            sentence = random.choice(IMP_list)
        else:
            sentence = random.choice(DEC_Q_list)
        sup_sub={4:0,3:0,5:0,6:1}
       
        
	

        if ch==1: # choice 1 for the IMP DEC business
            #print(g)
            #ns_list.append(weights[4])
            if "IMP" in sentence:
                sentence=sentence.replace("IMP","DEC")
            	#print(sentence)
	    sup_sub_k=sup_sub.keys()
            #random.shuffle(sup_sub_k)
            dir=grammar[:]
            for i in sup_sub_k:
                if ((grammar[0:i] + "0" + grammar[i+1:]) in LD.keys()) and ((grammar[0:i] + "1" + grammar[i+1:]) in LD.keys()):
                # print("gram:",grammar)
                    if (sentence in LD[grammar[0:i] + "0" + grammar[i+1:]]) and (
                             sentence in LD[grammar[0:i] + "1" + grammar[i+1:]]):
			#grammar = grammar[0:i] + str(sup_sub[i]) + grammar[i+1:]
			dir = dir[0:i] + str(sup_sub[i]) + dir[i+1:]

            if sentence in LD[grammar]: # checks if sentence is parsed by current grammar
                #print(grammar)
                for j in range(0,len(grammar)): # adjusts weights
                    #if grammar[j] == '0':
	            if dir[j]=='0':
                        if j == 6:
                            weights[j] = weights[j] - R * weights[j]
                        else:
                            weights[j] = weights[j] - r * weights[j]
                    else:
                        if j == 4 or j == 3 or j == 5:
                            weights[j] = weights[j] + R * (1 - weights[j])
                        else:
                            weights[j] = weights[j] + r * (1 - weights[j])
        else: # choice 2 for vanilla language domain

            #print(grammar)
            #if "IMP" in sentence:
                #sentence=sentence.replace("IMP","DEC")
	    sup_sub_k=sup_sub.keys()
            #random.shuffle(sup_sub_k)
            dir=grammar[:]
            for i in sup_sub_k:
                if ((grammar[0:i] + "0" + grammar[i+1:]) in LD.keys()) and ((grammar[0:i] + "1" + grammar[i+1:]) in LD.keys()):
                # print("gram:",grammar)
                    if (sentence in LD[grammar[0:i] + "0" + grammar[i+1:]]) and (
                             sentence in LD[grammar[0:i] + "1" + grammar[i+1:]]):
                        #grammar = grammar[0:i] + str(sup_sub[i]) + grammar[i+1:]
			dir = dir[0:i] + str(sup_sub[i]) + dir[i+1:]

            if sentence in LD[grammar]: # checks if sentence is parsed by current grammar
                #print(grammar)
                for j in range(0,len(grammar)): # adjusts weights
                    #if grammar[j] == '0':
		    if dir[j]=='0':
                        if j == 6:
                            weights[j] = weights[j] - R * weights[j]
                        else:
                            weights[j] = weights[j] - r * weights[j]
                    else:
                        if j == 4 or j == 3 or j == 5:
                            weights[j] = weights[j] + R * (1 - weights[j])
                        else:
                            weights[j] = weights[j] + r * (1 - weights[j])
        #writer.writerow(sentence)
        #print(weights[4])
        ns_list.append(weights[4])
    #print(ns_list)
    q.put(ns_list)  # this is not in loop hence will only be printed once.
    #results.append(ns_list)
    #return weights
def main():
    global LD
    jobs = []
    global results
    #sup_sub={4:1, 5:1}
    #languages = ['0001100000000']
    n = 0
    #results=[]
    q=Queue()
    numLearners = 100
    languages = ['0001101100011']*numLearners

    max_sentences = 10000
    print(max_sentences)
    #Gtarg = "0001001100011"
    reader=csv.reader(colag_file, delimiter='\t')
    for row in reader:
        LD[row[0]] = set()  # initializes the grammars as empty set of sentence
    ####
    #languages=LD.keys()    
    colag_file.close()
    file_again = open("COLAG_2011_flat_formatted.txt")
    # same thing with reader1 just for library reads the csv file as a csv reader object
    reader1 = csv.reader(file_again, delimiter='\t')
    for row1 in reader1:
        LD[row1[0]].add(row1[1] + row1[2])
        # q = Queue()

    results=[]
    # R=0.02
    ns_list = []
    n=0
    growth=2
    param=[]
    if growth==1: 
	if numLearners==1:
    		with open('linear_param.csv', 'rb') as f:
    			reader = csv.reader(f)
    			param = list(reader)
        else:
		param=[['2.12734126318465e-07', '-0.6782866308905113']]

    else:
	if numLearners==1:
		with open('e_param.csv', 'rb') as f:
    			reader = csv.reader(f)
    			param = list(reader)
	else:
		param=[['5516232.935663383', '9.936188693152848e-07']]
    
    print(languages)

    while n<len(languages):
        for i in range(numLearners):
	    if n>=len(languages):
                break
            IMP_list = []
            DEC_Q_list = []
            Gtarg=languages[n]
            for sent in LD[Gtarg]:
                if "IMP" in sent:
                    IMP_list.append(sent)
                elif "DEC" or "Q" in sent:
                    DEC_Q_list.append(sent)

            #pool = multiprocessing.Pool(processes=46)
            #print(DEC_Q_list)
            #for i in range(numLearners):
                #pool.apply_async(var_learner, args=(0.0004, 0.02, IMP_list, DEC_Q_list, max_sentences, Gtarg))
            #pool.close()
            #pool.join()
            #pool = mp.Pool(processes=4)
            
            p = multiprocessing.Process(target=var_learner, args=(0.001,0.2,IMP_list,DEC_Q_list,max_sentences,Gtarg,q,float(param[n][0]),float(param[n][1]),growth))
            n=n+1
            #print(n)
            jobs.append(p)
            p.start()
            #results.append(q.get())
        while 1:
            running = any(p.is_alive() for p in jobs)
            while not q.empty():
                results.append(q.get())
            if not running:
                break
    res=map(list, zip(*results))
    if growth==1:
    	with open("output_vl_linear4.csv", "wb") as f:
       		writer = csv.writer(f)
       		writer.writerows(res)
    else:
	with open("output_vl_e4.csv", "wb") as f:
       		writer = csv.writer(f)
       		writer.writerows(res)

    # print(result_list)
    #print(results)
    #print(np.array(results).shape)
    '''
    ns_list = np.zeros_like(results[0])
    #n_max_list=[-1]*max_sentences
    #n_min_list=[50000]*max_sentences

    
    #for result in results:
        #ns_list += np.array(result)
	#for i in range(0,max_sentences):
	    #if n_max_list[i]<result[i]:
		#n_max_list[i]=result[i]
	    #if n_min_list[i]>result[i]:
		#n_min_list[i]=result[i]

    x = np.arange(0, len(ns_list))

    y = np.array(results[49])
    y_max=np.array(results[0])
    y_min=np.array(results[99])

    plt.xlabel("No. of utterances ($10^7$)")
    plt.ylabel("NS parameter weight")
    plt.plot(x, y, label="average")
    plt.plot(x,y_max,label='max',linestyle='dashed',c='b')
    plt.plot(x,y_min,label='min',linestyle='dashed',c='g')
    plt.axvline(x=4566082,linestyle='dashed',label='age=2;6',c='r')
    plt.axvline(x=5610392,linestyle='dashed',label='age=3;0',c='c')
    plt.axvline(x=6688032,linestyle='dashed',label='age=3;6',c='m')
    plt.axvline(x=7787891,linestyle='dashed',label='age=4;0',c='y')
    plt.legend()

    #plt.xlim([150000,500000])
    if growth==1:
    	plt.savefig('var_linear_avg4.png')
    else:
        plt.savefig('var_e_avg4.png')
    '''
if __name__ == '__main__':
    main()
