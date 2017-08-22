import xlrd

book = xlrd.open_workbook('1-2c.xls')

sheet1 = book.sheet_by_index(0)


#column1 = sheet1.cell(0, 3)
#column2 = sheet1.cell(1, 3) 

cellA1 = sheet1.cell(0, 0) 
cellA2 = sheet1.cell(1, 0) 
cellA3 = sheet1.cell(2, 0) 
cellB1 = sheet1.cell(0, 1) 
cellB2 = sheet1.cell(1, 1)
cellB3 = sheet1.cell(2, 1)

#print column1
#print column2

# print cellA1.value
# print cellA2.value
# print cellA3.value

# print cellB1.value
# print cellB2.value
# print cellB3.value

array = []
line = []
#print sheet1.ncols

for row_index in range(sheet1.nrows):
    for col_index in range(sheet1.ncols):
        val = sheet1.cell_value(rowx=row_index, colx=col_index)
        #print val
        #print type(val)
        if isinstance(val,float):
        	line.append(val)
    
    if len(line) != 0:
		array.append(line)
		line = []

array.reverse()

print len(array)

#while len(array) != 0:
	#print array.pop() 




    



