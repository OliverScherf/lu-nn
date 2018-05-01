from enum import Enum

def getLayersFromFile(fileName):
  with open('model.txt', 'r') as myfile:
    data=myfile.read()

  lines = data.split("\n")

  layers = []
  for i in range(len(lines)):
    if "type: " in lines[i]:
      if "Convolution" in lines[i]:
        nameLine = lines[i-1]
        layers.append(nameLine[8:len(nameLine) - 1])
  return layers
      



# class ParserState(Enum):
#   NONE = 0
#   NAMESTART = 1
#   BLOCKSTART = 2
#   SUBNAMESTART = 3
#   SUBNAMEVALPREP = 4
#   INVAL = 5
#   OUTOFVAL = 6
#   BLOCKEND = 7

# state = ParserState.NONE
# currentData = dict()
# currentData["currentTag"] = ""
# currentData["tags"] = dict()
# currentData["vals"] = dict()


# def processCharForState(state, char):
#   global currentData
#   if state == ParserState.NONE:
#     if char == "{":
#       currentData["tags"] = dict()
#       currentData["vals"] = dict()
#       return ParserState.BLOCKSTART
#   if state == ParserState.BLOCKSTART:
#     if char == "}":
#       return ParserState.BLOCKEND
#     elif char != "\n" and char != " " and char != "\t":
#       currentData["currentTag"] = char
#       return ParserState.SUBNAMESTART
#   if state == ParserState.SUBNAMESTART:
#     if char == ":":
#       return ParserState.SUBNAMEVALPREP
#     elif char == "}":
#       return ParserState.BLOCKSTART
#     else:
#       currentData["currentTag"] += char
#       return ParserState.SUBNAMESTART
#   if state == ParserState.SUBNAMEVALPREP:
#     if char == "\"":
#       currentData["vals"][currentData["currentTag"]] = ""
#       return ParserState.INVAL
#     return ParserState.SUBNAMEVALPREP
#   if state == ParserState.INVAL:
#     if char == "\"":
#       return ParserState.BLOCKSTART
#     else:
#       currentData["vals"][currentData["currentTag"]] += char
#       return ParserState.INVAL
#   return state


# layerNames = []
# for char in data:
#   state = processCharForState(state, char)
#   if state == ParserState.BLOCKEND:
#     state = ParserState.NONE
#     if "type" in currentData["vals"]:
#       print("checking type ", currentData["vals"]["type"])
#       if currentData["vals"]["type"] == "Convolution":
#         print("got type conv")
#         currentData["layerNames"].append(currentData["vals"]["name"])
# print("got layers", layerNames)