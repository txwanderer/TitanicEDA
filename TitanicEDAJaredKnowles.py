
# coding: utf-8

# In[1]:

import pandas as pd
get_ipython().magic(u'pylab inline')


# In[2]:

dframe = pd.read_csv("train.csv")


# **PassegnerID EDA**
# 
# Is PassengerID categorical or continuous?

# In[3]:

dframe.PassengerId.value_counts()


# PassengerID is continuous.  A unique Passenger ID assigned to a passenger.

# Are there null values for PassengerID?

# In[4]:

dframe[dframe.PassengerId.isnull()]


# There are no null values for PassengerID

# Minimum value for Passenger ID

# In[5]:

dframe.PassengerId.min()


# Maximum value for PassengerID

# In[6]:

dframe.PassengerId.max()


# Mean for PassengerID

# In[7]:

dframe.PassengerId.mean()


# Standard Deviation for PassengerID

# In[8]:

dframe.PassengerId.std()


# Histogram for PassengerId

# In[9]:

dframe.PassengerId.hist()


# ***Summary for PassengerId***
# 
# <li> Is a continuous variable
# <li> There are no missing values for PassengerId
# <li> Minimum value for PassengerId is 1
# <li> Maximum value for PassengerId is 891
# <li> Mean for PassengerId is 446
# <li> Standard Deviation is 257.3538420152301
# <li> Each PassengerId from 1 to 891 was assigned to one person, or better to say that each passenger was assigned a unique ID

# ***Survived EDA***

# Is Survived continuous or categorical? 

# In[10]:

dframe.Survived.value_counts()


# Appears that Survived is categorized as a boolean: 0 is "not survived" and 1 is "survived"

# Are there any missing values for Survived?

# In[11]:

dframe[dframe.Survived.isnull()]


# There are no missing values for Survived.

# In[12]:

dframe.Survived.value_counts().plot(kind='bar')


# #### ***Summary for Survived***
# 
# <li> Survived is a categorial variable, where 0 is "did not survive" and 1 is "survived"
# <li> 549 passengers did not survive
# <li> 342 passengers did survive
# <li> There are no missing values for Survived

# ***Pclass EDA***
# 
# Is Pclass continuous or categorical?

# In[13]:

dframe.Pclass.value_counts()


# Pclass is categorical - Describes First Class (1), Second/Middle Class (2), and Third/Low Class (3)

# Are there any missing values for Pclass?

# In[14]:

dframe[dframe.Pclass.isnull()]


# There are no missing values for Pclass.

# In[15]:

dframe.Pclass.value_counts().plot(kind='bar')


# ***Summary for Pclass***
# 
# <li> Pclass is categorical:  1 is First Class, 2 is Middle/Second Class, 3 is Low/Third Class
# <li> There are no missing values for Pclass
# <li> 216 First Class Passengers, 184 Second Class Passengers, and 491 Third Class Passengers
# <li> Appears that the highest category population is Third Class, then first, and finally second

# *** Sex EDA***
# 
# Is Sex categorical or continuous?

# In[16]:

dframe.Sex.value_counts()


# Sex is categorical:  male and female categories
# 
# Are there any missing values?

# In[17]:

dframe[dframe.Sex.isnull()]


# There are no missing values.

# In[18]:

dframe.Sex.value_counts().plot(kind='bar')


# ***Summary for Sex***
# 
# <li> Sex is categorical:  Male or Female
# <li> There are no missing values for Sex
# <li> 577 male passengers and 314 female passengers

# *** EDA for SibSp***
# 
# Is it categorical or continuous?

# In[19]:

dframe.SibSp.value_counts()


# SibSp is continuous

# Missing values?

# In[20]:

dframe[dframe.SibSp.isnull()]


# There are no missing values for SibSp.

# Minimum value

# In[21]:

dframe.SibSp.min()


# Maximum value

# In[22]:

dframe.SibSp.max()


# Mean value

# In[23]:

dframe.SibSp.mean()


# Standard deviation

# In[24]:

dframe.SibSp.std()


# In[25]:

dframe.SibSp.hist(bins=9)


# *** Summary for SibSp ***
# 
# <li> SibSp (number of siblings and spouse on board) is continuous
# <li> There are no missing values
# <li> Minimum value is 0 Siblings/spouse on board
# <li> Maximum value is 8 Siblings/spouse on board
# <li> Mean value is about 0.52
# <li> Standard Deviation is about 1.103
# <li> Appears that most of the passengers did not have a sibling or spouse on board

# *** EDA for Parch ***
# 
# Is Parch continuous or categorical?

# In[26]:

dframe.Parch.value_counts()


# Parch is continuous

# Are there any missing values?

# In[27]:

dframe[dframe.Parch.isnull()]


# There are no missing values for Parch.

# Minimum value for Parch

# In[28]:

dframe.Parch.min()


# Maximum value for Parch

# In[29]:

dframe.Parch.max()


# Mean value for Parch

# In[30]:

dframe.Parch.mean()


# Standard Deviation for Parch

# In[31]:

dframe.Parch.std()


# In[32]:

dframe.Parch.hist(bins=7)


# ***Summary for Parch***
# 
# <li> Parch is continuous (Parent/Child on board)
# <li> There are no missing values
# <li> The minimum value is 0
# <li> The maximum value is 6
# <li> The mean value is 0.38
# <li> The standard deviation is 0.806
# <li> Appears that most of the passengers did not travel with a parent or child

# *** Ticket EDA ***

# Is it continuous or categorical?

# In[33]:

dframe.Ticket.value_counts()


# Ticket appears continuous since ticket numbers change, however tickets do not follow an incremental numerical pattern. Letters and words are weaved into ticket. This leads to possibly having to group by prefixes, which could make this categorical in nature (tickets bought with this group of people who are together, or bought at this port, etc). Will group by: Numeric, Pre-fixed, and "Line" tickets to satisfy a preliminary look at the data.  Should actually focus on what the prefixes are, group by specific prefix, and see how that relates to other variables.

# Are there null values?

# In[34]:

dframe[dframe.Ticket.isnull()]


# There are no missing values.

# Numeric only tickets

# In[35]:

dframe[dframe.Ticket.str.match('[0-9]')].Ticket.count()


# Pre-fixed tickets

# In[36]:

dframe[dframe.Ticket.str.match('[a-zA-Z].*[0-9]')].Ticket.count()


# "LINE" tickets

# In[37]:

dframe[dframe.Ticket.str.match('^[^0-9]*$')].Ticket.count()


# In[38]:

dframe.Ticket = dframe.Ticket.replace({'^[^0-9]*$' : 'LINE'}, regex=True)


# In[39]:

dframe.Ticket = dframe.Ticket.replace({'[a-zA-Z].*[0-9]': 'PREFIX'} , regex=True)


# In[40]:

dframe.Ticket = dframe.Ticket.replace({'^[0-9]*$': 'NUMERIC'} , regex=True)


# In[41]:

dframe


# In[42]:

dframe.Ticket.value_counts()


# In[43]:

dframe.Ticket.value_counts().plot(kind='bar')


# In[44]:

dframe[dframe.Ticket=='LINE']


# ***Note of Interest concerning Mr. William Henry Tornquist***
# After looking up his information on the encyclopedia titanica, it appears that he actually has a ticket number assigned to him - 370160.  More research is needed to correct this column if this is the first discrepancy to appear.

# In[45]:

dframe[dframe.Ticket=='PREFIX']


# *** Summary for Ticket ***
# 
# <li> Ticket initially appears to be continuous, but further analysis appears to point to this being categorical in nature:  there are numeric ticket numbers, ticket numbers with prefixes, and four tickets with LINE assigned for ticket number.  Seems that passengers are assigned to a ticket "group" of one or more passengers.
# <li> One of the "LINE" entries from the original csv file should be changed to an actual ticket number for Mr. William Henry Tornquist.  Might be other cases of incorrect ticket number values.
# <li> It appears that most passengers had a numerical style ticket number.

# *** EDA for Fare ***
# 
# Is it categorical or continuous?

# In[46]:

dframe.Fare.value_counts()


# Fare is continous

# In[47]:

dframe[dframe.Fare.isnull()]


# There are no missing values for Fare.

# Minimum value for Fare

# In[48]:

dframe.Fare.min()


# Maximum value for Fare

# In[49]:

dframe.Fare.max()


# Mean for Fare

# In[50]:

dframe.Fare.mean()


# Standard Deviation for Fare

# In[51]:

dframe.Fare.std()


# In[52]:

dframe.Fare.hist(bins=5)


# *** Summary for Fare ***
# 
# <li> Fare is a continuous variable covering price of ticket
# <li> There are no missing values for Fare
# <li> Minimum value of 0.0
# <li> Maximum value of 512.32
# <li> Mean value of Fare is 32.20
# <li> Standard deviation is about 49.69
# <li> Appears that the majority of passengers paid for a ticket that is less than 100

# *** EDA for Age ***
# 
# Is Age categorical or continuous?

# In[53]:

dframe.Age.value_counts()


# Age is continuous

# In[54]:

dframe[dframe.Age.isnull()]


# There are a 177 missing age values

# In[55]:

dframe.Age.min()


# In[56]:

dframe.Age.max()


# In[57]:

dframe.Age.mean()


# In[58]:

dframe.Age.std()


# In[59]:

dframe.Age.hist(bins=10)


# *** Summary for Age ***
# 
# <li> Age is a continuous variable
# <li> There are 177 missing age values
# <li> The minimum age is 0.42, assuming this means about 5 months old
# <li> The maximum age is 80 years
# <li> The mean age is 29.69 years
# <li> The Standard Deviation is about 14.52 years
# <li> It appears that highest populated age group would be around 20 to 30 years old.

# *** EDA for Cabin ***
# 
# Is it continuous or categorical?

# In[60]:

dframe.Cabin.value_counts()


# Cabin appears to be categorical since the cabins are designated by deck (the letter), and room (the number). Also reviewed titanic deck map to understand the "pair/groups of cabins" instances.  It appears there are some cabin clusters on certain decks which in some instances groups (families, etc) would have been assigned to. Numbers appear to start from the front of the ship going towards the back (fore and aft).

# Are there missing values?

# In[61]:

dframe[dframe.Cabin.isnull()]


# There are 687 missing values for cabin.

# In[62]:

cabinFrame = dframe


# Will try replacing existing values with just deck assignment for simplicity

# In[63]:

cabinFrame.Cabin = dframe.Cabin.replace({'^A.*$': 'A', '^B.*$': 'B', '^C.*$': 'C', '^D.*$': 'D', '^E.*$': 'E', '^F.*$': 'F', '^G.*$': 'G'} , regex=True)


# In[64]:

cabinFrame


# In[65]:

cabinFrame.Cabin.value_counts()


# In[66]:

cabinFrame[cabinFrame.Cabin=='T']


# Cabin T is on the "boat deck" of the Titanic.

# In[67]:

cabinFrame.Cabin.value_counts().plot(kind='bar')


# *** Summary of Cabin ***
# 
# <li> Cabin appears to be categorical according to deck, more refined grouping could be derived from analyzing number.
# <li> There are 687 missing values for cabin.
# <li> From data available in the dataset, it appears that the highest population decks are C and B. However, this is not accounting for the large amount of missing cabin information for the other passengers.

# *** Embarked EDA ***
# 
# Is Embarked continuous or categorical?

# In[68]:

dframe.Embarked.value_counts()


# Are there missing values?

# In[69]:

dframe[dframe.Embarked.isnull()]


# Two passengers missing Embarked information.

# In[71]:

dframe.Embarked.value_counts().plot(kind='bar')


# Appears that the majority of passengers boarded at Southampton.

# *** Summary for Embarked ***
# <li> Embarked is categorical: S = Southampton, C = Cherbourg, Q = Queenstown
# <li> There are only two missing instances in the dataset
# <li> Appears that the majority of passengers boarded at Southampton

# In[ ]:



