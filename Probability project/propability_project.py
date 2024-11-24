import numpy as np
import random as r


deck1 = []
deck2 = []

pile = []


alwaysslap = False

if alwaysslap:
    p1_chance = .9
else:
    p1_chance = 0

extra = False # if counting ascending/descending

p1_wins = 0
p2_wins = 0

nextplayer = 1
playedfacecard = 0

turns = 0
maxturns = 10000
running = True

def setup():
    global pile, deck1, deck2, running, turns
    cards = [x for x in range(1, 14)] * 4 

    r.shuffle(cards)
    #print(cards)
    
    deck1 = cards[:26]
    deck2 = cards[26:]
    pile = []
    
    #print("deck1: " + str(deck1))
    #print("deck2: " + str(deck2))
    running = True
    
    turns = 0

def p1Turn():
    #print("p1's turn")
    global playedfacecard
    
    if len(pile) == 0:
        playedfacecard = 0
    
    if running:
        playCard(1)

    if len(pile) != 0:
        playedfacecard = checkFaceCard()
    else:
        playedfacecard = 0

def p2Turn():
    #print("p2's turn")
    global playedfacecard
    
    if len(pile) == 0:
        playedfacecard = 0
    
    if running:
        playCard(2)
    
    if len(pile) != 0:
        playedfacecard = checkFaceCard()
    else:
        playedfacecard = 0

def playCard(player):
    global pile, nextplayer, playedfacecard
    #print(deck1)
    #print(deck2)
    #print("!!!!!!! " + str(pile) + " !!!!!!!")
    if player == 1:
        pile.insert(0, deck1.pop(0))
    if player == 2:
        pile.insert(0, deck2.pop(0))
    
    slappable = checkSlap()
    facecard = checkFaceCard()
    if alwaysslap:
        slapPile(player, slappable)
    else:
        if slappable:
            slapPile(player, slappable)
    
    if not slappable and running:
        if player == 1:
            nextplayer = 2
            if len(deck1) == 0:# and facecard == 0:
                #p1 loses
                #print("p1 loses")
                #print("in playcard")
                finish(2)
        if player == 2:
            nextplayer = 1
            if len(deck2) == 0:# and facecard == 0:
                #p2 loses
                #print("p2 loses")
                #print("in playcard")
                finish(1)
        if playedfacecard != 0 and running:
            if player == 1:
                #pass
                playOnFaceCard(2)
            if player == 2:
                #pass
                playOnFaceCard(1)

def checkFaceCard():
    if len(pile) != 0:
        if pile[0] == 11:
            return 1
        if pile[0] == 12:
            return 2
        if pile[0] == 13:
            return 3
        if pile[0] == 1:
            return 4
        return 0
    return 0


def playOnFaceCard(player):
    global pile, deck1, deck2, nextplayer
    #print(str(player) + " must play on face card " + str(playedfacecard))
    if player == 1:
        for c in range(playedfacecard):
            pile.insert(0, deck1.pop(0))
            slappable = checkSlap()
            if alwaysslap:
                slapPile(player, slappable)
            else:
                if slappable:
                    slapPile(player, slappable)
            facecard = checkFaceCard()
            if facecard != 0:
                #aslskjfhakjddhflkjahdf
                pass
            if facecard == 0 and c == playedfacecard - 1:
                #print("2 gets pile")
                deck2 += pile[::-1]
                #print(pile)
                pile = []
                nextplayer = 2
                turnIncrement()
            if len(deck1) == 0:
                break
        if len(deck1) == 0 and running:
            #print("p1 loses")
            #print("in playonfacecard")
            finish(2)
            
    if player == 2:
        for c in range(playedfacecard):
            #print("p2 plays card " + str(c))
            pile.insert(0, deck2.pop(0))
            slappable = checkSlap()
            if alwaysslap:
                slapPile(player, slappable)
            else:
                if slappable:
                    slapPile(player, slappable)
            facecard = checkFaceCard()
            if facecard != 0:
                #aslskjfhakjddhflkjahdf
                pass
            if facecard == 0 and c == playedfacecard - 1:
                #print("1 gets pile")
                deck1 += pile[::-1]
                #print(pile)
                pile = []
                nextplayer = 1
                turnIncrement()
            if len(deck2) == 0:
                break
        if len(deck2) == 0 and running:
            #print("p2 loses")
            #print("in playonfacecard")
            finish(1)
        

def slapPile(player, slappable):
    global pile, deck1, deck2, nextplayer, playedfacecard
    #print("pile: " + str(pile))
    num = r.random()
    if slappable:
        if num < p1_chance:
            #print("1 gets pile")
            deck1 += pile[::-1]
            #print(pile)
            pile = []
            playedfacecard = 0
            if len(deck2) == 0:
                #print("p2 loses")
                #print("in slappile slappable")
                finish(1)
            nextplayer = 1
            turnIncrement()
            #print(len(deck1))
            #print(len(deck2))
            #print("deck1: " + str(deck1))
            #print("deck2: " + str(deck2))
        else:
            #print("2 gets pile")
            deck2 += pile[::-1]
            #print(pile)
            pile = []
            playedfacecard = 0
            if len(deck1) == 0:
                #print("p1 loses")
                #print("in slappile slappable")
                finish(2)
            nextplayer = 2
            turnIncrement()
            #print(len(deck1))
            #print(len(deck2))
            #print("deck1: " + str(deck1))
            #print("deck2: " + str(deck2))
    if not slappable:
        if player == 1:
            if len(deck1) == 0:
                #print("p1 loses")
                #print("in slappile not slappable")
                finish(2)
            else:
                nextplayer = 2
                pile.append(deck1.pop(0))
        if player == 2:
            if len(deck2) == 0:
                #print("p2 loses")
                #print("in slappile not slappable")
                finish(1)
            else:
                nextplayer = 1
                pile.append(deck2.pop(0))
    

def checkSlap():
    if len(pile) > 1:
        if pile[0] == pile[1]:
            return True
    if len(pile) > 2:
        if pile[0] == pile[2]:
            return True
        if pile[0] == pile[1] + 1 and pile[1] == pile[2] + 1 and extra:
            return True
        if pile[0] == pile[1] - 1 and pile[1] == pile[2] - 1 and extra:
            return True
    return False

def turnIncrement():
    global turns
    turns += 1

def finish(winner):
    global running, p1_wins, p2_wins
    running = False
    if winner == 1:
        p1_wins += 1
    if winner == 2:
        p2_wins += 1
    

def turnLoop():
    global turns
    if nextplayer == 1 and running:
        p1Turn()
        turnIncrement()
    if nextplayer == 2 and running:
        p2Turn()
        turnIncrement()

for runs in range(100000):
    print(runs)
    setup()
    while turns < maxturns and running:
        turnLoop()
    if turns >= maxturns:
        print("No winners!")
        

print("p1_wins " + str(p1_wins))
print("p2_wins " + str(p2_wins))


#print("deck1: " + str(deck1))
#print("deck2: " + str(deck2))

