import sys
import os
import numpy as np
import pickle

LINHAS  = 3
COLUNAS = 3


class Estado:
  def __init__(self, p1, p2):
    self.tabuleiro = np.zeros((LINHAS, COLUNAS))
    self.p1 = p1
    self.p2 = p2
    self.acabou = False
    self.tabuleiroHash = None
    self.vezDe = 1 # o p1 começa

  # transforma o estado do tabuleiro em um string --> hash
  def getHash(self):
    self.tabuleiroHash = str(self.tabuleiro.reshape(COLUNAS * LINHAS))
    return self.tabuleiroHash

  # verifica se alguém ganhou ou deu empate
  def winner(self):
    # percorre as linhas
    for i in range(LINHAS):
      if sum(self.tabuleiro[i, :]) == 3:
        self.acabou = True
        return 1
      if sum(self.tabuleiro[i, :]) == -3:
        self.acabou = True
        return -1
    # percorre as colunas
    for i in range(COLUNAS):
      if sum(self.tabuleiro[:, i]) == 3:
        self.acabou = True
        return 1
      if sum(self.tabuleiro[:, i]) == -3:
        self.acabou = True
        return -1
    # percorre as diagonais
    diag_sum1 = sum([self.tabuleiro[i, i] for i in range(COLUNAS)])
    diag_sum2 = sum([self.tabuleiro[i, COLUNAS - i - 1] for i in range(COLUNAS)])
    diag_sum = max(abs(diag_sum1), abs(diag_sum2))
    if diag_sum == 3:
      self.acabou = True
      if diag_sum1 == 3 or diag_sum2 == 3:
        return 1
      else:
        return -1

    # empate
    # no available positions
    if len(self.jogadasPossiveis()) == 0:
      self.acabou = True
      return 0
    # not end
    self.acabou = False
    return None

  def jogadasPossiveis(self):
    positions = []
    for i in range(LINHAS):
      for j in range(COLUNAS):
        if self.tabuleiro[i, j] == 0:
          positions.append((i, j))
    return positions

  def updateEstado(self, position):
    self.tabuleiro[position] = self.vezDe
    # muda a vez para o outro jogador
    self.vezDe = -1 if self.vezDe == 1 else 1

  # only when game ends
  def daRecompensa(self):
    result = self.winner()

    if result == 1: # p1 ganhou
      self.p1.recompensar(1) # --> envia recompensa para p1
      self.p2.recompensar(0) # --> não recompensa p2
    elif result == -1: # p2 ganhou
      self.p1.recompensar(0) # --> não recompensa p1
      self.p2.recompensar(1) # --> envia recompensa para p2
    else: # empate
      self.p1.recompensar(0.1) # --> p1 recebe uma recompensa menor porque ele começou
      self.p2.recompensar(0.5) # --> p2 recebe uma recompensa maior porque ele jogou depois

  # reinicia o tabuleiro
  def reinicia(self):
    self.tabuleiro = np.zeros((LINHAS, COLUNAS))
    self.tabuleiroHash = None
    self.acabou = False
    self.vezDe = 1

  def treina(self, rounds=100): 
    for i in range(rounds):
      if i % 1000 == 0:
        print("Passos de treinamento {}".format(i))
      while not self.acabou:
        # É a vez do p1
        positions = self.jogadasPossiveis()
        p1_action = self.p1.escolherAcao(positions, self.tabuleiro, self.vezDe)
        # obtém a próxima ação e atualiza o tabuleiro
        self.updateEstado(p1_action)
        tabuleiro_hash = self.getHash()
        self.p1.addEstado(tabuleiro_hash)

        # verifica se o jogo acabou
        win = self.winner()
        if win is not None: # o jogo acabou
          self.daRecompensa() # dá as recompensas e reinicia o jogo
          self.p1.reinicia()
          self.p2.reinicia()
          self.reinicia()
          break
        else: # o jogo ainda não acabou
          # É a vez do p2
          positions = self.jogadasPossiveis()
          p2_action = self.p2.escolherAcao(positions, self.tabuleiro, self.vezDe)
          self.updateEstado(p2_action)
          tabuleiro_hash = self.getHash()
          self.p2.addEstado(tabuleiro_hash)

          win = self.winner()
          if win is not None:
            # self.exibeTabuleiro()
            # ended with p2 either win or draw
            self.daRecompensa()
            self.p1.reinicia()
            self.p2.reinicia()
            self.reinicia()
            break
  
  # IA vs Humano
  def joga(self):
    while not self.acabou:
      # Jogador 1
      positions = self.jogadasPossiveis()
      p1_action = self.p1.escolherAcao(positions, self.tabuleiro, self.vezDe)
      # take action and upate tabuleiro state
      self.updateEstado(p1_action)
      self.exibeTabuleiro()
      # check tabuleiro status if it is end
      win = self.winner()
      if win is not None:
        if win == 1:
          print(self.p1.nome, "venceu!")
        else:
          print("Empate!")
        self.reinicia()
        break
      else:
        # Jogador 2
        positions = self.jogadasPossiveis()
        p2_action = self.p2.escolherAcao(positions)

        self.updateEstado(p2_action)
        self.exibeTabuleiro()
        win = self.winner()
        if win is not None:
          if win == -1:
            print(self.p2.nome, "venceu!")
          else:
            print("Empate!")
          self.reinicia()
          break

  def exibeTabuleiro(self):
    # p1: x  p2: o
    for i in range(0, LINHAS):
      print('-------------')
      out = '| '
      for j in range(0, COLUNAS):
        if self.tabuleiro[i, j] == 1:
          token = 'x'
        if self.tabuleiro[i, j] == -1:
          token = 'o'
        if self.tabuleiro[i, j] == 0:
          token = ' '
        out += token + ' | '
      print(out)
    print('-------------')


class Jogador:
  def __init__(self, nome, exp_rate=0.3):
    self.nome = nome
    self.states = []  # registro de todos os estados
    self.lr = 0.2 # taxa de aprendizagem
    self.exp_rate = exp_rate # probabilidade de tomar uma ação aleatória
    self.decay_gamma = 0.9
    self.states_value = {}  # o valor de cada estado

  def getHash(self, tabuleiro):
    tabuleiroHash = str(tabuleiro.reshape(COLUNAS * LINHAS))
    return tabuleiroHash

  def escolherAcao(self, positions, current_tabuleiro, symbol):
    if np.random.uniform(0, 1) <= self.exp_rate:
      # realiza uma ação aleatória
      idx = np.random.choice(len(positions))
      action = positions[idx]
    else:
      value_max = -999
      for p in positions:
        next_tabuleiro = current_tabuleiro.copy() # cria uma cópia do tabuleiro
        next_tabuleiro[p] = symbol # simula a jogada
        next_tabuleiroHash = self.getHash(next_tabuleiro) # cria o hash
        value = 0 if self.states_value.get(next_tabuleiroHash) is None else self.states_value.get(next_tabuleiroHash)
        if value >= value_max:
          value_max = value
          action = p
    return action

  def addEstado(self, state):
    self.states.append(state)

  # Vamos entender o efeito da recompensa e como o valor dos estados é calculado:
  #   Todos os estados salvos são inicializados com o valor zero
  #   Depois seus valores são calculados da seguinte forma:
  #     valor(estado) += (taxa_de_aprendizagem x fator_de_desconto x recompensa - valor_anterior(estado))
  #   Exemplo:
  #     taxa_de_aprendizagem = 0.2
  #     fator_de_desconto = 0.9
  #     recompensa = 1
  #     Passo 0:
  #       valor(estado) = 0
  #     Passo 1:
  #       valor(estado) = 0 + 0.2 x (0.9 x 1.0 - 0) = 0.18
  def recompensar(self, reward):
    ganhou = reward == 1
    for st in reversed(self.states):
      if self.states_value.get(st) is None:
        self.states_value[st] = 0
      self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
      reward = self.states_value[st]
    if ganhou:
      print(len(self.states))
      for st in reversed(self.states):
        print(f'{st} - val: {self.states_value[st]}')
      sys.exit(0)


  def reinicia(self):
    self.states = [] # Limpa os estados, mas não o dicionário dos valores

  def salvar(self):
    fw = open('policy_' + str(self.nome), 'wb')
    pickle.dump(self.states_value, fw)
    fw.close()

  def carregar(self, file):
    fr = open(file, 'rb')
    self.states_value = pickle.load(fr)
    fr.close()

class JogadorHumano:
  def __init__(self, nome):
    self.nome = nome

  def escolherAcao(self, positions):
    while True:
      row = int(input("Input your action row:"))
      col = int(input("Input your action col:"))
      action = (row, col)
      if action in positions:
        return action

  def addEstado(self, state):
    pass

  def recompensar(self, reward):
    pass

  def reinicia(self):
    pass


if __name__ == "__main__":
  if not os.path.exists('policy_p1'):
    # training
    p1 = Jogador("p1")
    p2 = Jogador("p2")

    st = Estado(p1, p2)
    print("training...")
    st.treina(5000)
    p1.salvar()

  # play with human
  p1 = Jogador("computer", exp_rate=0)
  p1.carregar("policy_p1")

  p2 = JogadorHumano("human")

  st = Estado(p1, p2)
  st.joga()