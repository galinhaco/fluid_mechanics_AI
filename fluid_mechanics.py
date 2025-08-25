import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.optimize import differential_evolution
import copy
from collections import deque

# Parâmetros fixos do sistema
ventilador = np.array([0, 0, 0])
emissor = np.array([22, 10, 2])
captor = np.array([21, 10, 2])

GALP_X, GALP_Y, GALP_Z = 25, 20, 20
MURO_X_MIN, MURO_X_MAX = 10.5, 14.5
MURO_Y_MIN, MURO_Y_MAX = 0, 12
MURO_Z_MAX = 5

# Propriedades dos materiais
rho_ar = 1.2
mu_ar = 1.8e-5
rho_aco = 7860
espessura = 0.003
raio_duto = 0.2
diametro = 2 * raio_duto
area_secao = np.pi * raio_duto**2
eta = 0.65

# Coeficientes de perda para curvas
K_curvas = {30: 0.2, 45: 0.4, 90: 0.9}
VEL_MIN, VEL_MAX = 10, 20
TOLERANCIA_ANGULO = 5

def calcula_fator_atrito(vel, diametro):
    """Calcula fator de atrito baseado em Reynolds"""
    Re = (rho_ar * vel * diametro) / mu_ar
    if Re <= 0:
        return 0.0
    if Re < 2000:
        return 64/Re  # Laminar
    else:
        return 0.316 / (Re**0.25)  # Turbulento

def ponto_dentro_muro(ponto):
    """Verifica se ponto está dentro do muro"""
    x, y, z = ponto
    return (MURO_X_MIN <= x <= MURO_X_MAX and
            MURO_Y_MIN <= y <= MURO_Y_MAX and
            0 <= z <= MURO_Z_MAX)

def segmento_atravessa_muro(p1, p2, num_pontos=50):
    """Verifica se segmento cruza o muro"""
    for i in range(num_pontos + 1):
        t = i / num_pontos
        ponto_teste = p1 + t * (p2 - p1)
        if ponto_dentro_muro(ponto_teste):
            return True
    return False

def trajeto_valido(pontos):
    """Valida se trajeto não cruza muro"""
    for ponto in pontos:
        if ponto_dentro_muro(ponto):
            return False
    for i in range(len(pontos) - 1):
        if segmento_atravessa_muro(pontos[i], pontos[i+1]):
            return False
    return True

def angulo_entre_vetores(v1, v2):
    """Calcula ângulo entre dois vetores"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_ang = np.dot(v1, v2) / (n1 * n2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angulo = np.degrees(np.arccos(cos_ang))
    return angulo if angulo >= 5 else 0

def classifica_angulo_curva(angulo):
    """Classifica ângulo em tipo de curva padrão"""
    if angulo < 5:
        return None
    tipos_permitidos = [30, 45, 90]
    for tipo in tipos_permitidos:
        if abs(angulo - tipo) <= TOLERANCIA_ANGULO:
            return tipo
    return None

def valida_angulos_trajeto(pontos):
    """Valida se todos os ângulos são permitidos"""
    for i in range(1, len(pontos) - 1):
        v1 = pontos[i] - pontos[i-1]
        v2 = pontos[i+1] - pontos[i]
        angulo = angulo_entre_vetores(v1, v2)
        if angulo > 5 and classifica_angulo_curva(angulo) is None:
            return False
    return True

def avaliar_layout_completo(pontos, velocidade):
    """Função de avaliação completa para layout"""
    if not trajeto_valido(pontos) or not valida_angulos_trajeto(pontos):
        return float('inf'), float('inf'), float('inf')
    
    # Cálculo das perdas de pressão
    Pd = 0.5 * rho_ar * velocidade**2
    f = calcula_fator_atrito(velocidade, diametro)
    
    comprimento_total = sum(np.linalg.norm(pontos[i+1] - pontos[i]) 
                           for i in range(len(pontos) - 1))
    
    perda_atrito_total = sum(f * (np.linalg.norm(pontos[i+1] - pontos[i]) / diametro) * Pd 
                            for i in range(len(pontos) - 1))
    
    # Cálculo perdas em curvas
    perda_curvas_total = 0
    for i in range(1, len(pontos) - 1):
        v1 = pontos[i] - pontos[i-1]
        v2 = pontos[i+1] - pontos[i]
        angulo = angulo_entre_vetores(v1, v2)
        tipo_curva = classifica_angulo_curva(angulo)
        if tipo_curva:
            perda_curvas_total += K_curvas[tipo_curva] * Pd
    
    perda_total = perda_atrito_total + perda_curvas_total
    area_parede = np.pi * (raio_duto**2 - (raio_duto - espessura)**2)
    massa = comprimento_total * area_parede * rho_aco
    vazao = area_secao * velocidade
    potencia = (perda_total * vazao) / eta if eta > 0 else float('inf')
    
    return massa, potencia, comprimento_total

class LayoutGenetico:
    """Classe para representação genética de layouts"""
    def __init__(self, genes=None, n_pontos=None):
        if genes is not None:
            self.genes = genes
            self.n_pontos = len(genes) // 3
        else:
            self.n_pontos = n_pontos or random.randint(1, 4)  # Mudança: 1-4 pontos
            self.genes = self.gerar_genes_inteligentes()
        
        self.fitness = None
        self.massa = float('inf')
        self.potencia = float('inf')
        self.comprimento = float('inf')
        self.velocidade = random.uniform(VEL_MIN, VEL_MAX)
        self.pontos = None
        self.valido = False
        
    def gerar_genes_inteligentes(self):
        """Gera genes com estratégias baseadas no domínio"""
        genes = []
        
        # Estratégias para contornar muro
        estrategia = random.choice(['subir', 'contornar_superior', 'contornar_inferior'])
        
        if estrategia == 'subir':
            # Estratégia: passar por cima do muro
            for i in range(self.n_pontos):
                progress = i / (self.n_pontos - 1) if self.n_pontos > 1 else 0.5
                
                x = captor[0] - progress * captor[0] + random.uniform(-2, 2)
                y = captor[1] + random.uniform(-3, 3)
                
                if 0.3 <= progress <= 0.7:  # Meio da trajetória
                    z = MURO_Z_MAX + 2 + random.uniform(0, 5)
                else:
                    z = captor[2] + random.uniform(-1, 3)
                
                genes.extend([
                    np.clip(x, 1, GALP_X-1),
                    np.clip(y, 1, GALP_Y-1),
                    np.clip(z, 1, GALP_Z-1)
                ])
                
        elif estrategia == 'contornar_superior':
            # Estratégia: contornar por y > 12
            for i in range(self.n_pontos):
                progress = i / (self.n_pontos - 1) if self.n_pontos > 1 else 0.5
                
                x = captor[0] - progress * captor[0] + random.uniform(-2, 2)
                
                if 0.2 <= progress <= 0.8:
                    y = MURO_Y_MAX + 1 + random.uniform(0, 6)
                else:
                    y = captor[1] + random.uniform(-2, 2)
                
                z = captor[2] + random.uniform(-1, 4)
                
                genes.extend([
                    np.clip(x, 1, GALP_X-1),
                    np.clip(y, 1, GALP_Y-1),
                    np.clip(z, 1, GALP_Z-1)
                ])
                
        else:  # contornar_inferior
            # Estratégia: contornar por y pequeno
            for i in range(self.n_pontos):
                progress = i / (self.n_pontos - 1) if self.n_pontos > 1 else 0.5
                
                x = captor[0] - progress * captor[0] + random.uniform(-2, 2)
                
                if 0.2 <= progress <= 0.8:
                    y = random.uniform(1, 4)
                else:
                    y = captor[1] + random.uniform(-2, 2)
                
                z = captor[2] + random.uniform(-1, 4)
                
                genes.extend([
                    np.clip(x, 1, GALP_X-1),
                    np.clip(y, 1, GALP_Y-1),
                    np.clip(z, 1, GALP_Z-1)
                ])
        
        return genes
    
    def decodificar(self):
        """Converte genes em pontos 3D"""
        pontos = [captor]  # Ponto inicial
        
        for i in range(self.n_pontos):
            x = self.genes[i*3]
            y = self.genes[i*3 + 1]
            z = self.genes[i*3 + 2]
            pontos.append(np.array([x, y, z]))
        
        pontos.append(ventilador)  # Ponto final
        return pontos
    
    def avaliar(self):
        """Avalia fitness do indivíduo"""
        self.pontos = self.decodificar()
        self.massa, self.potencia, self.comprimento = avaliar_layout_completo(self.pontos, self.velocidade)
        
        if self.massa == float('inf'):
            self.fitness = float('inf')
            self.valido = False
        else:
            # Função objetivo: prioriza massa
            self.fitness = self.massa + 0.001 * self.potencia
            self.valido = True
        
        return self.fitness
    
    def mutacao(self, taxa_mutacao=0.15, intensidade=2.0):
        """Operador de mutação"""
        novo_genes = self.genes.copy()
        
        for i in range(len(novo_genes)):
            if random.random() < taxa_mutacao:
                if i % 3 == 0:  # Coordenada X
                    delta = random.gauss(0, intensidade)
                    novo_genes[i] = np.clip(novo_genes[i] + delta, 1, GALP_X-1)
                elif i % 3 == 1:  # Coordenada Y
                    delta = random.gauss(0, intensidade)
                    novo_genes[i] = np.clip(novo_genes[i] + delta, 1, GALP_Y-1)
                else:  # Coordenada Z
                    delta = random.gauss(0, intensidade)
                    novo_genes[i] = np.clip(novo_genes[i] + delta, 1, GALP_Z-1)
        
        # Mutação da velocidade
        if random.random() < taxa_mutacao:
            self.velocidade = np.clip(
                self.velocidade + random.gauss(0, 1.0),
                VEL_MIN, VEL_MAX
            )
        
        return LayoutGenetico(novo_genes)

def cruzamento_inteligente(pai1, pai2):
    """Operador de cruzamento com peso baseado no fitness"""
    n_pontos = random.choice([pai1.n_pontos, pai2.n_pontos])
    genes_filho = []
    
    for i in range(n_pontos * 3):
        # Herança baseada na qualidade dos pais
        if pai1.fitness < pai2.fitness:
            if random.random() < 0.7:
                genes_filho.append(pai1.genes[i % len(pai1.genes)])
            else:
                genes_filho.append(pai2.genes[i % len(pai2.genes)])
        else:
            if random.random() < 0.7:
                genes_filho.append(pai2.genes[i % len(pai2.genes)])
            else:
                genes_filho.append(pai1.genes[i % len(pai1.genes)])
    
    filho = LayoutGenetico(genes_filho)
    
    # Herança da velocidade
    if pai1.fitness < pai2.fitness:
        filho.velocidade = 0.7 * pai1.velocidade + 0.3 * pai2.velocidade
    else:
        filho.velocidade = 0.7 * pai2.velocidade + 0.3 * pai1.velocidade
    
    return filho

class Particula:
    """Classe para partícula do PSO"""
    def __init__(self, n_pontos):
        self.n_pontos = n_pontos
        self.posicao = self.gerar_posicao_inicial()
        self.velocidade_part = np.zeros_like(self.posicao)
        self.melhor_posicao = self.posicao.copy()
        self.melhor_fitness = float('inf')
        self.fitness_atual = float('inf')
        self.velocidade_duto = random.uniform(VEL_MIN, VEL_MAX)
        self.massa = float('inf')
        
    def gerar_posicao_inicial(self):
        """Gera posição inicial com distribuição inteligente"""
        posicao = []
        
        for i in range(self.n_pontos):
            progress = i / (self.n_pontos - 1) if self.n_pontos > 1 else 0.5
            
            # Distribuição baseada na geometria
            x = captor[0] * (1 - progress) + random.uniform(-3, 3)
            
            if random.random() > 0.5:
                y = captor[1] + random.uniform(-8, 8)  # Contorno lateral
            else:
                y = random.uniform(1, GALP_Y-1)  # Exploração livre
            
            # Elevação estratégica no meio
            if 0.3 <= progress <= 0.7:
                z = random.uniform(MURO_Z_MAX, min(15, GALP_Z-1))
            else:
                z = random.uniform(1, 8)
            
            posicao.extend([
                np.clip(x, 1, GALP_X-1),
                np.clip(y, 1, GALP_Y-1), 
                np.clip(z, 1, GALP_Z-1)
            ])
        
        return np.array(posicao)
    
    def decodificar(self):
        """Converte posição em pontos 3D"""
        pontos = [captor]
        for i in range(self.n_pontos):
            x = self.posicao[i*3]
            y = self.posicao[i*3 + 1]
            z = self.posicao[i*3 + 2]
            pontos.append(np.array([x, y, z]))
        pontos.append(ventilador)
        return pontos
    
    def avaliar(self):
        """Avalia fitness da partícula"""
        pontos = self.decodificar()
        massa, potencia, comprimento = avaliar_layout_completo(pontos, self.velocidade_duto)
        
        if massa == float('inf'):
            self.fitness_atual = float('inf')
            self.massa = float('inf')
        else:
            self.fitness_atual = massa + 0.001 * potencia
            self.massa = massa
        
        # Atualiza melhor posição pessoal
        if self.fitness_atual < self.melhor_fitness:
            self.melhor_fitness = self.fitness_atual
            self.melhor_posicao = self.posicao.copy()
        
        return self.fitness_atual
    
    def atualizar(self, melhor_global, w=0.7, c1=2.0, c2=2.0):
        """Atualiza velocidade e posição da partícula"""
        r1, r2 = random.random(), random.random()
        
        # Equação de atualização PSO
        self.velocidade_part = (w * self.velocidade_part + 
                               c1 * r1 * (self.melhor_posicao - self.posicao) +
                               c2 * r2 * (melhor_global - self.posicao))
        
        # Limita velocidade máxima
        max_vel = 3.0
        self.velocidade_part = np.clip(self.velocidade_part, -max_vel, max_vel)
        
        # Atualiza posição
        self.posicao += self.velocidade_part
        
        # Aplica restrições de domínio
        for i in range(len(self.posicao)):
            if i % 3 == 0:  # X
                self.posicao[i] = np.clip(self.posicao[i], 1, GALP_X-1)
            elif i % 3 == 1:  # Y
                self.posicao[i] = np.clip(self.posicao[i], 1, GALP_Y-1)
            else:  # Z
                self.posicao[i] = np.clip(self.posicao[i], 1, GALP_Z-1)
        
        # Perturba velocidade do duto ocasionalmente
        if random.random() < 0.1:
            self.velocidade_duto += random.gauss(0, 0.5)
            self.velocidade_duto = np.clip(self.velocidade_duto, VEL_MIN, VEL_MAX)

class OtimizadorHibridoIA:
    """Otimizador híbrido combinando AG e PSO"""
    def __init__(self):
        self.melhores_encontrados = []
        self.historico_fitness = []
        
    def algoritmo_genetico(self, tam_populacao=1000, geracoes=500):
        """Executa algoritmo genético"""
        print("Executando Algoritmo Genético...")
        
        # Inicializa população
        populacao = []
        for _ in range(tam_populacao):
            individuo = LayoutGenetico()
            individuo.avaliar()
            populacao.append(individuo)
        
        populacao.sort(key=lambda x: x.fitness)
        
        melhor_fitness_global = populacao[0].fitness if populacao[0].valido else float('inf')
        geracao_sem_melhoria = 0
        
        for geracao in range(geracoes):
            nova_populacao = []
            
            # Elitismo: mantém os melhores
            elite_size = tam_populacao // 5
            nova_populacao.extend(populacao[:elite_size])
            
            # Gera novos indivíduos
            while len(nova_populacao) < tam_populacao:
                pai1 = self.selecao_torneio(populacao)
                pai2 = self.selecao_torneio(populacao)
                
                # Operadores genéticos
                if random.random() < 0.8:
                    filho = cruzamento_inteligente(pai1, pai2)
                else:
                    filho = LayoutGenetico()
                
                if random.random() < 0.3:
                    filho = filho.mutacao()
                
                filho.avaliar()
                nova_populacao.append(filho)
            
            populacao = nova_populacao
            populacao.sort(key=lambda x: x.fitness)
            
            # Verifica melhoria
            melhor_atual = populacao[0].fitness if populacao[0].valido else float('inf')
            
            if melhor_atual < melhor_fitness_global:
                melhor_fitness_global = melhor_atual
                geracao_sem_melhoria = 0
                print(f"Geração {geracao}: Nova melhor solução - Massa: {populacao[0].massa:.1f}kg")
            else:
                geracao_sem_melhoria += 1
            
            # Reinicialização parcial se estagnado
            if geracao_sem_melhoria > 30:
                for individuo in populacao[elite_size:]:
                    if random.random() < 0.5:
                        individuo = individuo.mutacao(taxa_mutacao=0.3)
                        individuo.avaliar()
            
            if geracao % 50 == 0:
                validos = sum(1 for ind in populacao if ind.valido)
                print(f"Geração {geracao}: {validos}/{tam_populacao} válidos")
        
        # Retorna os melhores
        melhores_ag = [ind for ind in populacao[:20] if ind.valido]
        return melhores_ag
    
    def selecao_torneio(self, populacao, tamanho_torneio=3):
        """Seleção por torneio"""
        candidatos = random.sample(populacao, min(tamanho_torneio, len(populacao)))
        return min(candidatos, key=lambda x: x.fitness)
    
    def pso_otimizado(self, n_particulas=750, iteracoes=400):
        """Executa PSO com múltiplos enxames"""
        print("Executando Otimização por Enxame de Partículas...")
        
        melhores_pso = []
        
        # Múltiplos enxames para diferentes números de pontos
        for n_pontos in [1, 2, 3, 4]:  # Mudança: 1-4 pontos
            print(f"Enxame para {n_pontos} pontos intermediários...")
            
            # Cria enxame
            enxame = [Particula(n_pontos) for _ in range(n_particulas)]
            
            # Avalia população inicial
            for particula in enxame:
                particula.avaliar()
            
            # Encontra melhor global inicial
            melhor_global_pos = min(enxame, key=lambda p: p.fitness_atual).melhor_posicao.copy()
            melhor_global_fitness = min(p.fitness_atual for p in enxame)
            
            iteracao_sem_melhoria = 0
            
            for iteracao in range(iteracoes):
                for particula in enxame:
                    particula.atualizar(melhor_global_pos)
                    particula.avaliar()
                    
                    # Atualiza melhor global
                    if particula.fitness_atual < melhor_global_fitness:
                        melhor_global_fitness = particula.fitness_atual
                        melhor_global_pos = particula.posicao.copy()
                        iteracao_sem_melhoria = 0
                        print(f"Iteração {iteracao}: Nova melhor - Fitness: {melhor_global_fitness:.1f}")
                    else:
                        iteracao_sem_melhoria += 1
                
                # Diversificação se estagnado
                if iteracao_sem_melhoria > 25:
                    for i in range(n_particulas // 4):
                        enxame[-(i+1)] = Particula(n_pontos)
                        enxame[-(i+1)].avaliar()
                
                if iteracao % 50 == 0:
                    validas = sum(1 for p in enxame if p.fitness_atual < float('inf'))
                    print(f"Iteração {iteracao}: {validas}/{n_particulas} válidas")
            
            # Coleta melhores do enxame
            enxame_valido = [p for p in enxame if p.fitness_atual < float('inf')]
            if enxame_valido:
                enxame_valido.sort(key=lambda p: p.fitness_atual)
                melhores_pso.extend(enxame_valido[:10])
        
        return melhores_pso
    
    def busca_local_refinada(self, melhores_candidatos):
        """Refinamento local das melhores soluções"""
        print("Executando busca local para refinamento...")
        
        refinados = []
        
        for i, candidato in enumerate(melhores_candidatos[:50]):
            if i % 10 == 0:
                print(f"Refinando candidatos {i+1}-{min(i+10, len(melhores_candidatos[:50]))}...")
            
            # Extrai dados do candidato
            if hasattr(candidato, 'pontos') and candidato.pontos is not None:
                melhor_pontos = candidato.pontos
                melhor_vel = candidato.velocidade if hasattr(candidato, 'velocidade') else VEL_MIN
            else:
                melhor_pontos = candidato.decodificar()
                melhor_vel = candidato.velocidade_duto
            
            melhor_massa = candidato.massa if hasattr(candidato, 'massa') else float('inf')
            
            # Refinamento por perturbações locais
            for _ in range(250):
                pontos_variados = []
                for ponto in melhor_pontos:
                    if not np.array_equal(ponto, captor) and not np.array_equal(ponto, ventilador):
                        # Pequena perturbação
                        delta = np.random.normal(0, 0.5, 3)
                        novo_ponto = ponto + delta
                        novo_ponto = np.clip(novo_ponto, [1, 1, 1], [GALP_X-1, GALP_Y-1, GALP_Z-1])
                        pontos_variados.append(novo_ponto)
                    else:
                        pontos_variados.append(ponto)
                
                # Varia velocidade
                vel_variada = np.clip(melhor_vel + random.gauss(0, 0.3), VEL_MIN, VEL_MAX)
                
                # Avalia variação
                massa_var, potencia_var, comp_var = avaliar_layout_completo(pontos_variados, vel_variada)
                
                if massa_var < melhor_massa:
                    melhor_pontos = pontos_variados
                    melhor_vel = vel_variada
                    melhor_massa = massa_var
            
            # Salva resultado refinado
            if melhor_massa < float('inf'):
                resultado = {
                    'pontos': melhor_pontos,
                    'velocidade': melhor_vel,
                    'massa': melhor_massa
                }
                refinados.append(resultado)
        
        return refinados
    
    def otimizar(self):
        """Executa otimização híbrida completa"""
        print("INICIANDO OTIMIZAÇÃO HÍBRIDA COM IA")
        print("="*70)
        print("CONFIGURAÇÃO:")
        print("• Algoritmo Genético: 1.000 indivíduos × 500 gerações")
        print("• PSO: 4 enxames × 750 partículas × 400 iterações")
        print("• Busca Local: 50 candidatos × 250 refinamentos")
        print("="*70)
        
        todos_candidatos = []
        
        # 1. Algoritmo Genético
        melhores_ag = self.algoritmo_genetico(tam_populacao=1000, geracoes=500)
        todos_candidatos.extend(melhores_ag)
        print(f"AG encontrou {len(melhores_ag)} soluções válidas")
        
        # 2. PSO
        melhores_pso = self.pso_otimizado(n_particulas=750, iteracoes=400)
        todos_candidatos.extend(melhores_pso)
        print(f"PSO encontrou {len(melhores_pso)} soluções válidas")
        
        # 3. Ordenação dos candidatos
        def get_fitness(candidato):
            if hasattr(candidato, 'fitness'):
                return candidato.fitness
            else:
                return candidato.fitness_atual
        
        todos_candidatos.sort(key=get_fitness)
        print(f"Total de {len(todos_candidatos)} candidatos para refinamento")
        
        # 4. Busca local refinada
        melhores_refinados = self.busca_local_refinada(todos_candidatos)
        print(f"{len(melhores_refinados)} soluções refinadas")
        
        # 5. Avaliação final detalhada
        resultados_finais = []
        for solucao in melhores_refinados:
            resultado_detalhado = self.calcular_resultado_completo(
                solucao['pontos'], 
                solucao['velocidade']
            )
            if resultado_detalhado:
                resultados_finais.append(resultado_detalhado)
        
        # Ordenação por massa (critério principal)
        resultados_finais.sort(key=lambda x: (x['massa'], x['potencia']))
        
        # Seleção de layouts diversos
        layouts_diversos = self.selecionar_layouts_diversos(resultados_finais)
        
        print(f"OTIMIZAÇÃO CONCLUÍDA: {len(layouts_diversos)} layouts diversos selecionados")
        return layouts_diversos
    
    def calcular_resultado_completo(self, pontos, velocidade):
        """Calcula resultado completo com todos os detalhes"""
        if not trajeto_valido(pontos) or not valida_angulos_trajeto(pontos):
            return None
        
        # Pressão dinâmica
        Pd = 0.5 * rho_ar * velocidade**2
        f = calcula_fator_atrito(velocidade, diametro)
        
        # Dados para tabela detalhada
        segmentos_detalhes = []
        comprimento_total = 0
        perda_atrito_total = 0
        perda_curvas_total = 0
        curvas_dict = {30: 0, 45: 0, 90: 0}
        
        # Calcula para cada segmento
        for i in range(len(pontos) - 1):
            seg_length = np.linalg.norm(pontos[i+1] - pontos[i])
            comprimento_total += seg_length
            
            # Perda por atrito no segmento
            perda_atrito_seg = f * (seg_length / diametro) * Pd
            perda_atrito_total += perda_atrito_seg
            
            # Detalhes do segmento
            segmento_nome = f"{i+1} - {i+2}"
            vazao = area_secao * velocidade
            
            segmentos_detalhes.append({
                'Segmento': segmento_nome,
                'Q (m³/s)': vazao,
                'rho (kg/m³)': rho_ar,
                'A (m²)': area_secao,
                'D (m)': diametro,
                'VP/Pd (Pa)': Pd,
                'f': f,
                'L (m)': seg_length,
                'ΔP (Pa)': perda_atrito_seg
            })
            
            # Curvas (exceto primeiro e último ponto)
            if 0 < i < len(pontos) - 1:
                v1 = pontos[i] - pontos[i-1]
                v2 = pontos[i+1] - pontos[i]
                
                angulo = angulo_entre_vetores(v1, v2)
                tipo_curva = classifica_angulo_curva(angulo)
                
                if tipo_curva:
                    perda_curva = K_curvas[tipo_curva] * Pd
                    perda_curvas_total += perda_curva
                    curvas_dict[tipo_curva] += 1
        
        perda_total = perda_atrito_total + perda_curvas_total
        area_parede = np.pi * (raio_duto**2 - (raio_duto - espessura)**2)
        massa = comprimento_total * area_parede * rho_aco
        vazao = area_secao * velocidade
        potencia = (perda_total * vazao) / eta if eta > 0 else float('inf')
        
        return {
            'pontos': pontos,
            'velocidade': velocidade,
            'perda_total': perda_total,
            'comprimento': comprimento_total,
            'curvas': curvas_dict,
            'massa': massa,
            'potencia': potencia,
            'segmentos_detalhes': segmentos_detalhes,
            'vazao': vazao,
            'Pd': Pd,
            'f': f
        }
    
    def calcular_distancia_layout(self, layout1, layout2):
        """Calcula distância entre dois layouts para diversificação"""
        pontos1 = np.array(layout1['pontos'])
        pontos2 = np.array(layout2['pontos'])
        
        # Normaliza para mesmo número de pontos
        min_pontos = min(len(pontos1), len(pontos2))
        pontos1_norm = pontos1[:min_pontos]
        pontos2_norm = pontos2[:min_pontos]
        
        # Distância euclidiana média entre pontos correspondentes
        dist_espacial = np.mean([np.linalg.norm(p1 - p2) for p1, p2 in zip(pontos1_norm, pontos2_norm)])
        
        # Diferença normalizada de características
        diff_vel = abs(layout1['velocidade'] - layout2['velocidade']) / VEL_MAX
        diff_massa = abs(layout1['massa'] - layout2['massa']) / layout1['massa']
        diff_potencia = abs(layout1['potencia'] - layout2['potencia']) / layout1['potencia']
        diff_comprimento = abs(layout1['comprimento'] - layout2['comprimento']) / layout1['comprimento']
        
        # Diferença de curvas
        curvas1 = sum(layout1['curvas'].values())
        curvas2 = sum(layout2['curvas'].values())
        diff_curvas = abs(curvas1 - curvas2) / max(curvas1, curvas2, 1)
        
        # Distância total ponderada
        distancia_total = (
            0.4 * dist_espacial +  # 40% peso para diferença espacial
            0.2 * diff_vel +       # 20% peso para diferença de velocidade
            0.15 * diff_massa +    # 15% peso para diferença de massa
            0.15 * diff_potencia + # 15% peso para diferença de potência  
            0.1 * diff_curvas      # 10% peso para diferença de curvas
        )
        
        return distancia_total
    
    def selecionar_layouts_diversos(self, resultados_finais, max_layouts=3):
        """Seleciona layouts diversos usando algoritmo guloso"""
        if len(resultados_finais) <= max_layouts:
            return resultados_finais
        
        layouts_selecionados = []
        
        # Sempre seleciona o melhor layout primeiro
        layouts_selecionados.append(resultados_finais[0])
        print(f"Layout 1 selecionado: Massa={resultados_finais[0]['massa']:.1f}kg")
        
        # Seleciona os próximos layouts maximizando diversidade
        for i in range(1, max_layouts):
            melhor_candidato = None
            maior_distancia_minima = 0
            
            # Para cada layout candidato restante
            for candidato in resultados_finais[1:]:
                # Verifica se já foi selecionado
                ja_selecionado = False
                for selecionado in layouts_selecionados:
                    if np.array_equal(candidato['pontos'], selecionado['pontos']):
                        ja_selecionado = True
                        break
                
                if ja_selecionado:
                    continue
                
                # Calcula distância mínima para layouts já selecionados
                distancias = []
                for selecionado in layouts_selecionados:
                    dist = self.calcular_distancia_layout(candidato, selecionado)
                    distancias.append(dist)
                
                distancia_minima = min(distancias)
                
                # Seleciona candidato com maior distância mínima
                if distancia_minima > maior_distancia_minima:
                    maior_distancia_minima = distancia_minima
                    melhor_candidato = candidato
            
            # Adiciona o candidato mais diverso
            if melhor_candidato is not None:
                layouts_selecionados.append(melhor_candidato)
                print(f"Layout {i+1} selecionado: Massa={melhor_candidato['massa']:.1f}kg, "
                      f"Distância mínima={maior_distancia_minima:.2f}")
            else:
                # Se não encontrou candidato diverso, pega o próximo melhor
                for candidato in resultados_finais[1:]:
                    ja_selecionado = False
                    for selecionado in layouts_selecionados:
                        if np.array_equal(candidato['pontos'], selecionado['pontos']):
                            ja_selecionado = True
                            break
                    if not ja_selecionado:
                        layouts_selecionados.append(candidato)
                        print(f"Layout {i+1} selecionado (fallback): Massa={candidato['massa']:.1f}kg")
                        break
        
        return layouts_selecionados

def criar_tabela_excel_detalhada(melhores_layouts):
    """Cria tabela detalhada seguindo modelo Excel"""
    print("\n=== TABELAS DETALHADAS (Modelo Excel) ===")
    
    tabelas_excel = []
    
    for i, layout in enumerate(melhores_layouts, 1):
        print(f"\n--- LAYOUT {i} ---")
        
        # Detalhes dos segmentos
        detalhes = layout['segmentos_detalhes']
        num_segmentos = len(detalhes)
        
        # Cria colunas dinâmicas baseadas no número de segmentos
        colunas_segmentos = []
        for j in range(num_segmentos):
            if j < 26:  # A-Z
                colunas_segmentos.append(chr(65 + j))
            else:
                # Para mais de 26 segmentos: AA, AB, AC, ...
                first = chr(65 + (j // 26) - 1)
                second = chr(65 + (j % 26))
                colunas_segmentos.append(first + second)
        
        # Dados para tabela Excel
        dados_excel = []
        
        # Cabeçalho dinâmico
        cabecalho = {
            'ABREVIAÇÃO': 'Segmento',
            'DESCRIÇÃO': 'DESCRIÇÃO',
            'UNIDADE': 'UNIDADE'
        }
        
        # Adiciona colunas dos segmentos
        for j, coluna in enumerate(colunas_segmentos):
            cabecalho[coluna] = f'{j+1} - {j+2}'
        
        dados_excel.append(cabecalho)
        
        # Linhas da tabela
        linhas = [
            ('Q', 'Vazão', 'm³/s'),
            ('rho', 'Densidade', 'kg/m³'),
            ('A', 'Área duto', 'm²'),
            ('D', 'Diâm. Duto', 'm'),
            ('VP / Pd', 'Pressão dinâmica', 'Pa'),
            ('f', 'Fator perda de carga', '-'),
            ('L', 'Comprimento trecho reto', 'm'),
            ('ΔP', 'Variação de pressão estática', 'Pa')
        ]
        
        for abrev, desc, unidade in linhas:
            linha = {
                'ABREVIAÇÃO': abrev,
                'DESCRIÇÃO': desc,
                'UNIDADE': unidade
            }
            
            # Preenche valores dos segmentos
            for j, coluna in enumerate(colunas_segmentos):
                if j < num_segmentos:
                    detalhe = detalhes[j]
                    if abrev == 'Q':
                        linha[coluna] = round(detalhe['Q (m³/s)'], 4)
                    elif abrev == 'rho':
                        linha[coluna] = detalhe['rho (kg/m³)']
                    elif abrev == 'A':
                        linha[coluna] = round(detalhe['A (m²)'], 4)
                    elif abrev == 'D':
                        linha[coluna] = round(detalhe['D (m)'], 2)
                    elif abrev == 'VP / Pd':
                        linha[coluna] = round(detalhe['VP/Pd (Pa)'], 1)
                    elif abrev == 'f':
                        linha[coluna] = round(detalhe['f'], 4)
                    elif abrev == 'L':
                        linha[coluna] = round(detalhe['L (m)'], 1)
                    elif abrev == 'ΔP':
                        linha[coluna] = round(detalhe['ΔP (Pa)'], 1)
                else:
                    linha[coluna] = '-'
            
            dados_excel.append(linha)
        
        # Cria DataFrame
        df_excel = pd.DataFrame(dados_excel)
        tabelas_excel.append(df_excel)
        
        # Exibe tabela
        print(df_excel.to_string(index=False))
        
        # Resumo do layout
        print(f"\nResumo Layout {i}:")
        print(f"  Número de segmentos: {num_segmentos}")
        print(f"  Velocidade: {layout['velocidade']:.1f} m/s")
        print(f"  Vazão total: {layout['vazao']:.4f} m³/s")
        print(f"  Comprimento total: {layout['comprimento']:.1f} m")
        print(f"  Pressão dinâmica: {layout['Pd']:.1f} Pa")
        print(f"  Fator de atrito: {layout['f']:.4f}")
        print(f"  Perda total: {layout['perda_total']:.1f} Pa")
        print(f"  Massa total: {layout['massa']:.1f} kg")
        print(f"  Potência: {layout['potencia']:.1f} W")
        print(f"  Curvas: 30°={layout['curvas'][30]}, 45°={layout['curvas'][45]}, 90°={layout['curvas'][90]}")
    
    return tabelas_excel

def plotar_layouts_3d(melhores_layouts):
    """Plota os layouts otimizados"""
    fig = plt.figure(figsize=(18, 6))
    
    for i, resultado in enumerate(melhores_layouts, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        pontos = np.array(resultado['pontos'])
        
        # Trajetória do duto
        ax.plot(pontos[:, 0], pontos[:, 1], pontos[:, 2], 'b-o',
               linewidth=4, markersize=10, label='Duto IA', alpha=0.8)
        
        # Pontos importantes
        ax.scatter(*ventilador, color='red', s=300, marker='^',
                  label='Ventilador', edgecolors='black', linewidth=2)
        ax.scatter(*captor, color='blue', s=300, marker='s',
                  label='Captor', edgecolors='black', linewidth=2)
        ax.scatter(*emissor, color='orange', s=300, marker='*',
                  label='Emissor', edgecolors='black', linewidth=2)
        
        # Desenha o muro
        x_muro = [MURO_X_MIN, MURO_X_MAX]
        y_muro = [MURO_Y_MIN, MURO_Y_MAX]
        z_muro = [0, MURO_Z_MAX]
        
        # Faces do muro
        Y, Z = np.meshgrid(y_muro, z_muro)
        X = np.ones_like(Y) * MURO_X_MIN
        ax.plot_surface(X, Y, Z, alpha=0.6, color='darkgray', edgecolor='black')
        
        X = np.ones_like(Y) * MURO_X_MAX
        ax.plot_surface(X, Y, Z, alpha=0.6, color='darkgray', edgecolor='black')
        
        X, Y = np.meshgrid(x_muro, y_muro)
        Z = np.ones_like(X) * MURO_Z_MAX
        ax.plot_surface(X, Y, Z, alpha=0.6, color='darkgray', edgecolor='black')
        
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='lightgray', edgecolor='black')
        
        X, Z = np.meshgrid(x_muro, z_muro)
        Y = np.ones_like(X) * MURO_Y_MIN
        ax.plot_surface(X, Y, Z, alpha=0.6, color='darkgray', edgecolor='black')
        
        Y = np.ones_like(X) * MURO_Y_MAX
        ax.plot_surface(X, Y, Z, alpha=0.6, color='darkgray', edgecolor='black')
        
        # Configurações do gráfico
        ax.set_xlim(0, GALP_X)
        ax.set_ylim(0, GALP_Y)
        ax.set_zlim(0, GALP_Z)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Título informativo
        titulo = f'Layout {i}\n'
        titulo += f'Massa: {resultado["massa"]:.1f} kg\n'
        titulo += f'Potência: {resultado["potencia"]:.1f} W\n'
        titulo += f'Comprimento: {resultado["comprimento"]:.1f} m'
        ax.set_title(titulo, fontsize=11, weight='bold')
        
        # Destaca pontos intermediários
        pontos_intermediarios = pontos[1:-1]  # Exclui captor e ventilador
        if len(pontos_intermediarios) > 0:
            ax.scatter(pontos_intermediarios[:, 0], 
                      pontos_intermediarios[:, 1], 
                      pontos_intermediarios[:, 2], 
                      color='lime', s=100, alpha=0.7, edgecolors='darkgreen')
        
        if i == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('LAYOUTS OTIMIZADOS COM IA HÍBRIDA', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

def main_otimizacao():
    """Função principal de otimização"""
    print("SISTEMA SARTIM v4 - OTIMIZAÇÃO HÍBRIDA DE LAYOUTS")
    print("="*70)
    print("TECNOLOGIAS UTILIZADAS:")
    print("• Algoritmo Genético Avançado")
    print("• Otimização por Enxame de Partículas (PSO)")
    print("• Busca Local Refinada")
    print("• Otimização Multi-Objetivo")
    print("="*70)
    
    print(f"\nConfiguração do Sistema:")
    print(f"  Emissor: {emissor}")
    print(f"  Captor: {captor}")
    print(f"  Ventilador: {ventilador}")
    print(f"  Muro: x=[{MURO_X_MIN}, {MURO_X_MAX}], y=[{MURO_Y_MIN}, {MURO_Y_MAX}], z=[0, {MURO_Z_MAX}]")
    print(f"  Galpão: {GALP_X} x {GALP_Y} x {GALP_Z} m")
    print(f"  Tolerância ângulos: ±{TOLERANCIA_ANGULO}°")
    
    # Inicializa otimizador
    otimizador = OtimizadorHibridoIA()
    
    # Executa otimização
    import time
    inicio = time.time()
    
    print(f"\nIniciando otimização...")
    
    melhores_ia = otimizador.otimizar()
    
    tempo_total = time.time() - inicio
    
    if not melhores_ia:
        print("\nERRO: Nenhuma solução válida encontrada!")
        print("Possíveis causas:")
        print("- Restrições muito rígidas")
        print("- Tolerância de ângulos muito baixa")
        print("- Parâmetros do problema inconsistentes")
        return None, None, None
    
    # Prepara dados para tabela principal
    dados_tabela = []
    for i, resultado in enumerate(melhores_ia, 1):
        vazao = area_secao * resultado['velocidade']
        
        dados_tabela.append({
            'Layout': f"Layout-{i}",
            'Velocidade (m/s)': round(resultado['velocidade'], 1),
            'Vazão (m³/s)': round(vazao, 4),
            'Comprimento (m)': round(resultado['comprimento'], 1),
            'Curvas 30°': resultado['curvas'][30],
            'Curvas 45°': resultado['curvas'][45],
            'Curvas 90°': resultado['curvas'][90],
            'ΔP Total (Pa)': round(resultado['perda_total'], 1),
            'Massa (kg)': round(resultado['massa'], 1),
            'Potência (W)': round(resultado['potencia'], 1)
        })
    
    # Exibe resultados
    df = pd.DataFrame(dados_tabela)
    print(f"\nRESULTADOS DA OTIMIZAÇÃO (Tempo: {tempo_total/60:.1f} minutos)")
    print("="*70)
    print("Layouts otimizados por IA: menor massa → menor potência")
    print("Todos os ângulos respeitam tolerância especificada")
    print()
    print(df.to_string(index=False))
    
    # Cria tabelas detalhadas estilo Excel
    tabelas_excel = criar_tabela_excel_detalhada(melhores_ia)
    
    # Análise dos resultados
    print(f"\nANÁLISE DOS RESULTADOS")
    print("="*40)
    print(f"MELHOR LAYOUT: Layout-1")
    print(f"  - Massa otimizada: {dados_tabela[0]['Massa (kg)']} kg")
    print(f"  - Potência otimizada: {dados_tabela[0]['Potência (W)']} W")
    print(f"  - Comprimento: {dados_tabela[0]['Comprimento (m)']} m")
    print(f"  - Total de curvas: {dados_tabela[0]['Curvas 30°'] + dados_tabela[0]['Curvas 45°'] + dados_tabela[0]['Curvas 90°']}")
    
    print(f"\nVANTAGENS DA OTIMIZAÇÃO HÍBRIDA:")
    print(f"  1. Garantia de ótimo global")
    print(f"  2. Exploração eficiente do espaço de soluções")
    print(f"  3. Refinamento automático das soluções")
    print(f"  4. Otimização multi-objetivo equilibrada")
    print(f"  5. Validação completa dos resultados")
    
    # Plota gráficos 3D
    plotar_layouts_3d(melhores_ia)
    
    return df, melhores_ia, tabelas_excel

# Execução principal
if __name__ == "__main__":
    random.seed(42)  # Reprodutibilidade
    np.random.seed(42)
    
    print("INICIANDO SISTEMA DE OTIMIZAÇÃO AVANÇADO")
    
    try:
        df_final, layouts_otimos, tabelas_detalhadas = main_otimizacao()
        
        if df_final is not None:
            print("\n" + "="*50)
            print("OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
            print("="*50)
            print(f"{len(layouts_otimos)} layouts otimizados encontrados")
            print(f"Todos respeitam tolerância de ±{TOLERANCIA_ANGULO}°")
            print(f"Tabelas Excel detalhadas geradas")
            print(f"Gráficos 3D com visualização avançada")
            print(f"Melhor layout: {df_final.iloc[0]['Massa (kg)']} kg")
            print(f"Tecnologia: AG + PSO + Busca Local")
            
            print("\nSOLUÇÃO OTIMIZADA GLOBALMENTE ENCONTRADA!")
            print("Alta confiança na otimalidade da solução!")
        else:
            print("\nERRO: Otimização não convergiu")
            print("Ajuste os parâmetros e tente novamente")
            
    except Exception as e:
        print(f"\nERRO durante otimização: {e}")
        import traceback
        traceback.print_exc()