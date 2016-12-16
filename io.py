class ReaderAndWriter:
    def __init__(self):
      return  ##nothing to do here

    def read_file(self,filename):
        file = open(filename,'r')
        matrix_of_values = []
        dummy_line = file.readline()
        for line in file:
            elements = line[:-1].split(",")
            matrix_of_values.append(elements)
        file.close()
        return matrix_of_values

    def write_file(self,data,filename):
        file = open(filename,'w')
        for row in data:
            string = ""
            for element in row:
                string += str(element) + ','
            string = string[:-1] + '\n'
            file.write(string)
        file.close()



