# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:25:25 2019

@author: Tere

Description: create a new data-set, replaced the string-type as numeric-types.
Better to use logistic regression.
"""


origen=open("adult.data-header.csv","r")
destino=open("adult.data-header-renewtypes.csv","w")
cambios=open("renew-types-description.csv","w")

#read origen
Dataframe=[]
Datasets=[]
myset=set()

line=origen.readline()
for line in origen:
    row=[]
    row = line.split(",")
    for i in range(len(row)):
        row[i] = row[i].strip()
        data=row[i]
        if not(data.isnumeric()):
            myset.add(data)
    Dataframe.append(row)

origen.close()

listaset=(list(myset))
listaset.sort()

dicdata = {listaset[i-1]:i  for i in range(1,len(listaset)+1)}
dicdatainv = {i:listaset[i-1]  for i in range(1,len(listaset)+1)}

for line in Dataframe:
    s=""
    for item in line:
        if item.isnumeric():
            s += item + ","
        else:
            s += str(dicdata[item]) + ","            
    s += '\n'
    destino.write(s)
destino.close()

cambios.write("String-type,Numeric-value-in-file\n")
for item in listaset:
    cambios.write(str(item)+","+str(dicdata[item])+"\n")
cambios.close()
    