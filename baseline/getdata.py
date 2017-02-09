import os
import re
import magic
import imghdr
import PyPDF2
import time
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



path = ['/home/flipped/PycharmProject/baseline/data/', '/home/flipped/PycharmProject/baseline/data2/']

data = [[],[]]
stops = set(stopwords.words("english"))

#transform pdf into text file
for i in path:
    for filename in os.listdir(i):

        with open(os.path.join(i, filename), 'r') as f:
            #determine file type
            file_type = magic.from_buffer(f.read(1024), mime=True)
            print file_type
            if file_type == 'application/pdf':
                if not os.path.exists('pdfTOtext/'):
                    os.makedirs('pdfTOtext/')
                if os.path.exists('pdfTOtext/'):
                    path.append('pdfTOtext/')

                pdfReader = PyPDF2.PdfFileReader(f)
            #create file and extract pdf content into it
                file = open(os.path.join('pdfTOtext/', filename + ".txt"), 'w+')
                #file = open(filename + ".txt",'w+')
                for i in xrange(pdfReader.getNumPages()):
                    page = pdfReader.getPage(i)
                    page_content = page.extractText().encode('utf-8')
                    file.write(page_content)

                file.close()
            else:
                continue


#get contect from text file
for i in path:
    for filename in os.listdir(i):
        file_name, file_extension = os.path.splitext(filename)
        #print file_name
        if file_extension != '.pdf':
            with open(os.path.join(i, filename), 'r') as f:

                #f_type = magic.Magic(mime=True, uncompress=True)
                #file_type = magic.from_buffer(f.read(1024), mime=True)
                #if file_type != 'application/pdf':
                     # print f
                  document = f.read()
                  document = " ".join(filter(lambda x: x[0] != '@', document.split()))
                  #print document
                  #document = BeautifulSoup(document)
                  document = document.replace("http://", "")
                  document = document.replace("www.", "")
                  #print document
                  letters_only = re.sub("[^a-zA-Z]", " ", document)
                  letters_only = re.sub(" +", " ", letters_only)
                  words = ' '.join([word for word in letters_only.split() if word not in stops])

            data[0].append(filename)
            data[1].append(words)
        else:
            continue
        #print data[1]
        #if file_type == 'application/pdf':


x = input("Input your choice, 1: TfidfVectorizer, 2: CounterVectorizer, 3: LSI : ")

print("\nExtracting features...")
if   x == 1:
    y = input("Input feature numbers : ")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=y )
    # get vector matrix
    data_matrix = vectorizer.fit_transform(data[1])
    #data_matrix = data_matrix.toarray()
    print("  done in %.3fsec" % (time.time() - t0))
elif x == 2:
    t0 = time.time()
    y = input("Input feature numbers : ")
    vectorizer = CountVectorizer(max_features=y)
    # get vector matrix
    data_matrix = vectorizer.fit_transform(data[1])
    #data_matrix = data_matrix.toarray()
    print("  done in %.3fsec" % (time.time() - t0))
elif x == 3:
    t0 = time.time()
    y = input("Input feature numbers > 100: ")
    vectorizer = TfidfVectorizer(max_features=y)
    # get vector matrix
    data_matrix = vectorizer.fit_transform(data[1])
    print("\nPerforming dimensionality reduction using LSA...")
    svd = TruncatedSVD(n_components=100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    data_matrix = lsa.fit_transform(data_matrix)
    #data_matrix = data_matrix.toarray()
    #print data_matrix
    print("  done in %.3fsec" % (time.time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

print ("\nNumber of documents and features: \n" )
print data_matrix.shape
#get feature names
terms = vectorizer.get_feature_names()
print terms
#compute cosine distance
#dist = 1 - cosine_similarity(data_matrix)
#print dist

#Compute k-means
print ("\nInput the number of cluster centers, the number should not be larger than %d" % data_matrix.shape[0] )
num_cluster_centers = input('Input the number of cluster centers: ')
km = KMeans(n_clusters=num_cluster_centers, init='k-means++', max_iter=100, n_init=1)
km.fit(data_matrix)
clusters = km.labels_.tolist()

#print top words per cluster
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_cluster_centers):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :20]:
        print ' %s' % terms[ind],
    print



