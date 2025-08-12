dataset = [] #created list


with open('catdata.txt', 'r') as file :
    dataset=file.readlines()
    print(f'loaded {len(dataset) } lines')

