import spacy
sp = spacy.load('en_core_web_sm')

import math

words_dict={}
docs_dict={}


def title(titleText, docId):

    stop_chars = [",", "*", "\n", "\r\n", "\r", "'s", "-", ";", ":", "?", "(", ")", '"', "[", "]","–","." ] 
    stop_words=[",", "-", "/", "'s", "\r", "\r\n", "\n ", "\r\n ", "\n", ";", ":", "?", "(", ")", '"', "[", "]","–",'is', " ", "stack", "overflow", 'to', 'of', 'the', 'ourselves', 'hers', 'yourself', 'there', 'they', 'own', 'an', 'be', 'for', 'its', 'yours', 'such', 'into', 'of', 'itself', 'off', 'is', 's', 'am', 'or', 'as', 'from', 'him', 'each', 'the', 'themselves', 'are', 'we', 'these', 'your', 'his', 'don', 'nor', 'me', 'were', 'her', 'himself', 'this', 'our', 'their', 'to', 'ours', 'had', 'she', 'at', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'so', 'he', 'you', 'herself', 'has', 'myself',  'those', 'i', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'it', 'was', 'here', 'how', 'can']

    for i in stop_chars:
        titleText = titleText.replace(i, "")

    titleText=titleText.lower().split(" ")#it wiil be in the form of a list

    #To delete stopwords
    i=0;
    while(i<len(titleText)):
        if titleText[i] in stop_words or len(titleText[i])==0:
                del titleText[i]
                i=i-1
        i+=1

    #this is to lemmitization
    titleTextTemp=titleText[0]
    for i in range(1,len(titleText)):
        titleTextTemp+=" "+titleText[i]
    titleTextTemp = sp(titleTextTemp)
    titleText=[]

    #creating words dictionary with documents ids.
    for word in titleTextTemp:
        titleText.append(word.lemma_)
        try:
            words_dict[word.lemma_].append(docId)
        except:
            words_dict[word.lemma_]=[docId]


    #to store the documents into dictionary and score for each word
    for word in titleText:
        if docId in docs_dict:
            if word in docs_dict[docId]:
                docs_dict[docId][word]["tf"]=+1
            else:
                docs_dict[docId].update({word:{"tf":1}})
        else:
            docs_dict[docId]={word:{"tf":1}}


    #to calculate denominator mode score for each documents
    tempWord_dict={}
    for word in titleText:
        try:
            tempWord_dict[word]+=1
        except:
            tempWord_dict[word]=1
    docs_dict[docId]["denom-netor-score"]=CalcDenomMode(tempWord_dict, len(titleText))
    docs_dict[docId]["total-terms"]=len(titleText)




#denominator mode calculating function
def CalcDenomMode(dict, total):
    sqr = 0
    for term in dict:
        sqr = sqr+ (dict[term]/total)**2
    sqr = math.sqrt(sqr)
    return sqr
      

#calculating tf-idf value and associating it with for each token for each documents
def CalcTF_IDF(N):
    for id in docs_dict:
        for word in docs_dict[id]:
            try:
                df = len(words_dict[word])
                idf = math.log10(N/df)

                docs_dict[id][word]["tf_idf"]=((1+math.log10(docs_dict[id][word]["tf"]))/docs_dict[id]["total-terms"])*idf
            except:
                pass



def userQuery(titleText):
    
    stop_chars = [",", "*", "\n", "\r\n", "\r", "'s", "-", ";", ":", "?", "(", ")", '"', "[", "]","–","." ] 
    stop_words=[",", "-", "/", "'s", "\r", "\r\n", "\n ", "\r\n ", "\n", ";", ":", "?", "(", ")", '"', "[", "]","–",'is', " ", "stack", "overflow", 'to', 'of', 'the', 'ourselves', 'hers', 'yourself', 'there', 'they', 'own', 'an', 'be', 'for', 'its', 'yours', 'such', 'into', 'of', 'itself', 'off', 'is', 's', 'am', 'or', 'as', 'from', 'him', 'each', 'the', 'themselves', 'are', 'we', 'these', 'your', 'his', 'don', 'nor', 'me', 'were', 'her', 'himself', 'this', 'our', 'their', 'to', 'ours', 'had', 'she', 'at', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'so', 'he', 'you', 'herself', 'has', 'myself',  'those', 'i', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'it', 'was', 'here', 'how', 'can']

    for i in stop_chars:
        cleanText = titleText.replace(i, "")

    cleanText=cleanText.lower().split(" ")#it wiil be in the form of a list

    #To delete stopwords
    i=0;
    while(i<len(cleanText)):
        if cleanText[i] in stop_words or len(cleanText[i])==0:
                del cleanText[i]
                i=i-1
        i+=1

    #this is to lemmitization
    titleTextTemp=cleanText[0]
    for i in range(1,len(cleanText)):
        titleTextTemp+=" "+cleanText[i]
    titleTextTemp = sp(titleTextTemp)
    cleanText=[]

    #creating words dictionary with documents ids.
    length = len(titleTextTemp)
    clean_dict={}
    for word in titleTextTemp:
        cleanText.append(word.lemma_)
        try:
            clean_dict[word.lemma_]+=1/length
        except:
            clean_dict[word.lemma_]=1/length
    
    # print(cleanText)
    return {"cleantext":cleanText, "cleandict":clean_dict}


#getting all the documents in which the words are present
def get_Docs(cleanInp, words_dict_file):
    docs_list=[]
    for word in cleanInp:
        try:
            docs_list=docs_list+words_dict_file[word]
        except:
            pass
    
    return set(docs_list)




import json
#These codes are for indexing :----
train=0
if train:
    data=""
    with open('sof20k.json') as f:
        data = json.load(f)

    print(len(data))
    j=0
    for i in data:
        # if(j>=10):
        #     break
        title(i["title"], i["id"])
        j+=1
    id = len(data)
    CalcTF_IDF(id)

    # print(docs_dict)


    with open('sof_words_dict.json', 'w+') as f:
        json.dump(words_dict, f)

    with open('sof_docs_dict.json', 'w+') as f:
        json.dump(docs_dict, f)

words_dict_file={}
docs_dict_file={}
with open('sof_docs_dict.json') as f:
  docs_dict_file = json.load(f)

with open('sof_words_dict.json') as f:
  words_dict_file = json.load(f)


inp =input("Enter the input\n")
cleanTextList=userQuery(inp)
docs_list_from_words_dict=get_Docs(cleanTextList["cleantext"], words_dict_file)
docs_list_dict={}#to store all the documents in which at least one word is present

# print(docs_list_from_words_dict)

for docid in docs_list_from_words_dict:
    docs_list_dict.update({docid:docs_dict_file[str(docid)]})

# print(docs_list_dict)
print(cleanTextList["cleandict"])

final_ranking={}
query_denom_score=CalcDenomMode(cleanTextList["cleandict"], len(cleanTextList["cleantext"]))
# print("denom is ",query_denom_score)
for id in docs_list_dict:
    score=0
    for word in cleanTextList["cleandict"]:
        try:
            score = score + cleanTextList["cleandict"][word]*docs_list_dict[id][word]["tf_idf"]
        except:
            pass
    final_ranking[id]=score/(query_denom_score*docs_list_dict[id]["denom-netor-score"])

final_ranking = sorted(final_ranking.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
# print(final_ranking[0])
for i in range(5):
    print(final_ranking[i])




