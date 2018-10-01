
# coding: utf-8

# In[9]:


#initial test function to check if programming logic works
#No matrix, just a simple function
def MED(n, m):
    
    if n == "":
        return len(m)
    if m == "":
        return len(n)
    if n[-1]==m[-1]:
        cost = 0
    elif n[-1]!=m[-1]:
        cost = 2
    else:
        cost = 1
    dist = min (MED(n[:-1],m)+1, MED(n, m[:-1])+1, MED(n[:-1], m[:-1])+cost)
    return dist
                                                         
                                                         


# In[18]:


def MED_Matrix(n, m):
    
    rows = len(n)+1                                       #initializing rows using the length of the source string
    cols = len(m)+1                                       #initializing columns using length of target string
    dist = [[0 for x in range(cols)] for x in range(rows)]#initializing the calculator for distance using imbedded for statements, basically(rows*columns)
    for i in range(1, rows):            
        dist[i][0] = i                                    #base case (), when target is empty, delete all source and get total cost
    for i in range(1, cols):
        dist[0][i] = i                                    #base case (), when source is empty, insert characters into target using target as template
        
    for col in range(1, cols):
        for row in range(1, rows):
            if n[row-1] == m[col-1]:                     #for case characters are the same, no cost incurred
                cost = 0
            else:
                cost = 2                                    #declaring initial cost of substitution
                
            #using recursive function, compute matrix whilst keeping track of cost
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    #print out the matrix, though clumsily
    for r in range(rows):
        print(dist[r])
    
    #return the min distance as computed above
    return dist[row][col]


# In[19]:


print (MED_Matrix("intention", "execution"))

