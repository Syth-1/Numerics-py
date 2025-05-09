from enum import Enum
from typing import Union

Number = Union[int, float]


class staticproperty(staticmethod):
    def __get__(self, *_):         
        return self.__func__()



class Matrix2String(): 
    @staticmethod
    def to_string(arr : list) -> str: 
        spacer = [0 for i in arr[0]]
        
        for i2 in range(len(arr)): 
            for i1 in range(len(arr[0])):
                spacer[i1] = max(spacer[i1], len(str(arr[i2][i1])))

        out_string = " | "
        for i2 in range(len(arr)):
            for i1 in range(len(arr[0])): 
                obj = str(arr[i2][i1])
                for i in range(spacer[i1] - len(obj)):
                    out_string += " "
                out_string += obj
                if (i1 < len(arr[0])-1):
                    out_string += ", "   
            out_string += "\n"

        out_string = out_string[:-1].replace("\n", " |\n | ")+" |"
        return out_string