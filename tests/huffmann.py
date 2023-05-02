#!/usr/bin/python3

#drawn from https://towardsdatascience.com/huffman-encoding-python-implementation-8448c3654328

class Node:
    def __init__(self, prob, symbol, left=None, right=None) -> None:
        #probability of symbol
        self.prob = prob
        
        #the symbol itself
        self.symbol = symbol

        #the left Node
        self.left = left

        #the right Node
        self.right = right

        #tree direction: 0 I am on the left of the parent Node, 1 I am on the right. The value of the incoming edge
        self.code = ''

"""A helper function to calculate the probailities of symbols in given data"""
def Calculate_Probabilities(data):
    symbols = {} #initalise a dict key=symbol value=frequency

    for element in data:
        if element not in symbols:
            symbols[element] = 1
        else:
            symbols[element] += 1

    return symbols

"""A helper function to store the codes by travelling the HuffmanTree"""
codes = {} #initialize a dict with all codes (key=symbol, value=the 1 and 0 suite (Huffman Code). It is a global variable
def CalculateNodes(node, val=''): #method called recursively val is the actual suite of 0 and 1 of the parent Node, node is the current node
    newval = val + node.code #the actual Huffman Code of node (the values of the edges arriving to node 0 if it is a left edge from the parent node 1 otherwiser)
    if (node.left):
        CalculateNodes(node.left, newval) #it is not a leaf node <==> It is not a symbol
    if (node.right):
        CalculateNodes(node.right, newval) #it is not a leaf node <==> It is not a symbol
    if (not node.left and not node.right):
        codes[node.symbol] = newval #it is a leaf node so a symbol we want to encode

    return codes #we return the global variable with its state

"""A helper function to obtain the encoded output
    data it the input (a string), coding is the global variable codes we contructed in the above function
    to remember about coding: the keys are the symbol the values are the corresponding huffman encoding (alse oa string)
"""
def Output_Encoded(data, coding): 
    encoding_output = []
    for c in data:
        #print(c, end='')
        encoding_output.append(coding[c]) 
    str_encoded = ''.join([str(code) for code in encoding_output])

    return str_encoded

"""A helper function to calculate the gain between compressed and not compressed data
    data it the input (a string), coding is the global variable codes we contructed in the above function
    to remember about coding: the keys are the symbol the values are the corresponding huffman encoding (also a string)
    coding's keys are all the symbol to be found in data
"""
def TotalGain(data, coding):
    before_compression = len(data) * 8
    after_compression = 0

    symbols = coding.keys() #symbols contains all symbols to be found in data
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += len(coding[symbol]) * count

    print(f"Before compression the size of {data} (in bits) is {before_compression}")
    print(f"After compression the size of {data} (in bits) has been reduced to {after_compression}")

"""The Main function the Huffmann encoding itself"""
def HuffmanEncoding(data):
    symbols_with_probs = Calculate_Probabilities(data)
    symbols = symbols_with_probs.keys()
    probabilities = symbols_with_probs.values()
    print(f"list of symbols: {symbols}")
    print(f"list of corresponding probabilities: {probabilities}")

    nodes = [] #the list of nodes we have to reduce to one root node
    #first step all symbols as nodes in a list (there are the leaf values)
    for symbol in symbols:
        nodes.append(Node(symbols_with_probs[symbol], symbol)) #the left and right ar by default null, code (string) is by default empty

    while (len(nodes) > 1): #we have to reduce the list of nodes to one root node
        #sort all nodes in the ascending order based on their probabilities
        ## see https://docs.python.org/fr/3/howto/sorting.html
        nodes = sorted(nodes, key=lambda node: node.prob)
        
        #to verify
        #print("les noeuds restant dans la liste")
        #for node in nodes:
            #print(f"+noeud symbole {node.symbol} de probabilité {node.prob}")

        #takes the two smallest nodes (relative to their probability)
        right = nodes[0] #the smallest
        right.code = str(1) #the value of the edge arriving to that node
        left = nodes[1] #the second smallest
        left.code = str(0) #the value of the edge arriving to that node
        ## and combine them
        ### the only important thing is the combined probaility (sum) not the combined symbol (that we don't use)
        new_node = Node(left.prob+right.prob, f"{left.symbol}-{right.symbol}", left=left, right=right)

        #updates the nodes list with the new_node replacing left and right nodes
        nodes.append(new_node)
        nodes.remove(left)
        nodes.remove(right)
    
    #nodes[0] is the only remaining node, the root node of all
    huffmann_encoding = CalculateNodes(nodes[0]) #val is empty by dafault as return we get the codes global dict with keys as symbols and values as huffmanencoding
    output_encoded = Output_Encoded(data, huffmann_encoding)
    #print()
    print(f"[Huffmann]:encoded output {output_encoded}")
    TotalGain(data, huffmann_encoding) #only prints to the console
    return output_encoded, nodes[0]

if __name__ == "__main__":
    data="bonjourtoutlemonde"
    print(f"[Main] Before encoding |{data}|")
    HuffmanEncoding(data)


