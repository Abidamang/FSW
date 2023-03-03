#!bin/python


a=0
print(bin(0))
b=format(a, 'b')

print(b)
 

# my_rnd_list =[0,2,5,8,4,9,3,1]
# largest = -1
# for item in my_rnd_list:
#     if (item > largest):
#         largest = item
#     print(largest, end=' ')
# print('')
# print('Largest Number is :', largest)

# initializing string 
# str_1 = "Join our freelance network"

# str_1_encoded = bytes(str_1,'UTF-8')

# #printing the encode string 
# print(str_1_encoded)

# for bytes in str_1_encoded:
#     print(bytes, end = ' ')
#=======================================================================================================

#1 - COVERT STRING TO LIST

# import json
# o_string='{"John":"01","Rick":"02","Dazy":"10"}'

# # Printing original
# print("The Original string : "+str(o_string))

# # Covert to dictionary
# result = json.loads(o_string)
# print(result)

# # Iterating through
# for i in result:
#     print(i)

#=======================================================================================================
# # 2 - CONVERT DIC TO LIST

# o_string={'John': 10, 'Eric': 20, 'Dazy':30 }

# #  Prining Dictionary
# print("This is simple string: "+str(o_string))

# resultList= list(o_string.items())

# # Printing convert dictionary to list
# print("Coverted to a list ",resultList)

# # Printing key
# resultList= list(o_string.keys())
# print("Key Key: ", resultList)

# # Printing value
# resultList= list(o_string.values())
# print("Key Value: ", resultList)

# # ZIP FUNCTIONS
# resultList= zip(o_string.keys(), o_string.values())
# # Returned Object
# print(resultList)
# # Convert object to a list
# resultList=list(resultList)
# # Printing List
# print(resultList)
#=======================================================================================================
# # 2 - CONVERT DIC TO LIST
o_string={'John': 10, 'Eric': 20, 'Dazy':30 }

# Create empty list
resultList = []

# Traversing through each key value pair of a dictionary using items() function keys

for key, val in o_string.items():
    
    # appending list of corresponding key value

    resultList.append([key, val])

# printing
print(resultList)





#Print Coverted to Dictionary
#print("The converted results :",str(result))

# result=dict((a.strip(), int(b.strip()))
#             for a,b in (element.split('-')
#                 for element in o_string.split('-')))
# print(result)