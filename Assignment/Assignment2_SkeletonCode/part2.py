import robby
#import numpy as np
from utils import *
import random
#import matplotlib.pyplot as plt
POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rw.graphicsOff()


def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    gene = ""
    for i in range(length):
        temp = random.randint(0, 6)
        gene = gene + str(temp)
    return gene



def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """


    """Your Code Here"""
    geneList = []
    for i in range(0,size):
        geneList.append(randomGenome(length))
    return geneList

def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    #print("genome:",genome)
    #print("length of genome:",len(genome))
    #print("type:",type(genome))
    if type(genome) is not str or len(genome) != 243:
        raise Exception("strategy is not a string of length 243")
    for char in genome:
        if char not in "0123456":
            raise Exception("strategy contains a bad character: '%s'" % char)
    if type(steps) is not int or steps < 1:
        raise Exception("steps must be an integer > 0")
    if type(init) is str:
        # init is a config file
        rw.load(init)
    elif type(init) in [int, float] and 0 <= init <= 1:
        # init is a can density
        rw.goto(0, 0)
        rw.distributeCans(init)
    else:
        raise Exception("invalid initial configuration")

    totalFitness=0
    average=0
    for i in range(0,25):
        rw.goto(0,0)
        rw.distributeCans(init)
        tempFitness=0
        for i in range(0,steps):
            percept = rw.getPerceptCode()
            move = genome[percept]
            if move=='0':
                tempFitness=tempFitness+rw.north()
            if move=='1':
                tempFitness=tempFitness+rw.south()
            if move=='2':
                tempFitness=tempFitness+rw.east()
            if move=='3':
                tempFitness=tempFitness+rw.west()
            if move=='4':
                tempFitness=tempFitness+rw.stay()
            if move=='5':
                tempFitness=tempFitness+rw.grab()
            if move=='6':
                tempFitness=tempFitness+rw.random()
        totalFitness=totalFitness+tempFitness
    avgFitness=totalFitness/25
    return avgFitness


def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    tempList = []
    for i in population:
        tempList.append(fitness(i))
    average = sum(tempList) / len(tempList)
    bestStrategy=population[tempList.index(max(tempList))]
    return average, max(tempList),bestStrategy


def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    cutPoint = random.randint(1, len(genome1) - 1)
    # print("--------------------------")
    # print("cut point is:",cutPoint)
    newGene1 = ""
    newGene2 = ""
    # print("length of genome1:",len(genome1))
    for i in range(0, len(genome1)):
        if i >= cutPoint:
            newGene1 = newGene1 + genome2[i]
            newGene2 = newGene2 + genome1[i]
        else:
            newGene1 = newGene1 + genome1[i]
            newGene2 = newGene2 + genome2[i]
    # print("after crossover, gene1:",newGene1)
    # print("after crossover, gene2:",newGene2)
    return newGene1, newGene2


def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    newGenome = ""
    for i in range(0, len(genome)):
        tempRand = random.random()
        if mutationRate > tempRand:
            if genome[i]=='0':
                newGenome = newGenome +str(random.choice([1,2,3,4,5,6]))
            elif genome[i]=='1':
                newGenome=newGenome+str(random.choice([0,2,3,4,5,6]))
            elif genome[i]=='2':
                newGenome=newGenome+str(random.choice([0,1,3,4,5,6]))
            elif genome[i]=='3':
                newGenome=newGenome+str(random.choice([0,1,2,4,5,6]))
            elif genome[i]=='4':
                newGenome=newGenome+str(random.choice([0,1,2,3,5,6]))
            elif genome[i]=='5':
                newGenome=newGenome+str(random.choice([0,1,2,3,4,6]))
            elif genome[i]=='6':
                newGenome=newGenome+str(random.choice([0,1,2,3,4,5]))
        else:
            newGenome = newGenome + genome[i]
    return newGenome



def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """
    fitnessList = []
    for i in range(1,len(population)+1):
        fitnessList.append(i)
    newGene1 = weightedChoice(population, fitnessList)
    newGene2 = weightedChoice(population, fitnessList)
    return newGene1, newGene2



def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    #print("Population size:", populationSize)
    #print("Genome length: 243")
    population = makePopulation(populationSize, 243)
    outputFile=open(logFile,'w')
    outputFile2=open("bestStrategy.txt",'w')
    wholeBestGener=0
    wholeBestAvg=0
    wholeBestFit=0
    wholeBestStra=""
    avgList=[]
    bestList=[]
    for i in range(0,300):
        mutatedList = []
        crossList = []
        countPopulation = 0
        population = sortByFitness(population)[0]
        '''while(len(mutatedList)!=populationSize):
            #population = sortByFitness(population)[0]
            selectedGene1, selectedGene2 = selectPair(population)
            #print("-----------------------")
            crossoverRand = random.random()
            if crossoverRate > crossoverRand:
                selectedGene1, selectedGene2 = crossover(selectedGene1, selectedGene2)
            #crossList.append(selectedGene1)
            #crossList.append(selectedGene2)
            selectedGene1=mutate(selectedGene1,mutationRate)
            selectedGene2=mutate(selectedGene2,mutationRate)
            mutatedList.append(selectedGene1)
            mutatedList.append(selectedGene2)'''
        while (len(crossList) != populationSize):
            #spopulation = sortByFitness(population)[0]
            selectedGene1, selectedGene2 = selectPair(population)
            crossoverRand = random.random()
            if crossoverRate > crossoverRand:
                selectedGene1, selectedGene2 = crossover(selectedGene1, selectedGene2)
            crossList.append(selectedGene1)
            crossList.append(selectedGene2)
        while(len(mutatedList)!=populationSize):
            tempMutate=mutate(crossList[countPopulation],mutationRate)
            mutatedList.append(tempMutate)
            countPopulation=countPopulation+1
        population = mutatedList
        avg, best, bestStrategy= evaluateFitness(mutatedList)
        avgList.append(avg)
        bestList.append(best)
        if wholeBestFit < best:
            wholeBestFit = best
            wholeBestStra = bestStrategy
            wholeBestGener = i
            wholeBestAvg = avg
        if i%10==0:
            avg, best,bestStrategy = evaluateFitness(mutatedList)
            print("Generation", i, ": average fitness", round(avg,2), ", best fitness:", best)
            outputFile.write(str(i) + " " + str(avg) + " " + str(best) + " "+str(bestStrategy)+"\n")
        else:
            continue
    '''
    #graph part, comment out if need to test the graph
    plt.plot(avgList, color='skyblue', label="Average Fitness")
    plt.plot(bestList, color='red', label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.yticks([])
    plt.legend(loc='best')
    plt.show()'''
    outputFile2.write(str(wholeBestGener) + " " + str(wholeBestAvg) + " " + str(wholeBestFit) + " "+str(wholeBestStra)+"\n")
    outputFile.close()
    outputFile2.close()


def test_FitnessFunction():
    f = fitness(rw.strategyM)
    print("Fitness for StrategyM : {0}".format(f))

test_FitnessFunction()
#pop=makePopulation(5,243)
#mutate(pop[1],0.01)
runGA(50, 0.5, 0.005,"GAoutput.txt​")