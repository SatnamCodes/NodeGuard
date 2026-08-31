import networkx as nx
import random 
import numpy as np 
random.seed(42)
np.random.seed(42)


NUM_NORMAL_ACCOUNTS = 300
NUM_MULE_ACCOUNTS_LAYERING = 8 #layering chain A->B->C
NUM_MULE_ACCOUNTS_FUNNEL = 6 # many to one , high betweenness
MULE_TRANSACTIONS_PER_HOP = 12
NORMAL_TRANSACTIONS_PER_ACCOUNT = 10

def generate_normal_accounts(graph,num_accounts):
    for account_id in range(num_accounts):
        graph.add_node(account_id,is_mule=False)


def add_normal_transactions(graph,num_accounts):
    for account_id in range(num_accounts):
        num_transactions = random.randint(1,NORMAL_TRANSACTIONS_PER_ACCOUNT)
        for _ in range (num_transactions):
            target = random.randint(0,num_accounts - 1)
            while target == account_id:
                target = random.randint(0,num_accounts-1)
            amount = round(random.uniform(100,5000),2)
            timestamp = random.randint(0,10000)
            graph.add_edge(account_id,target,amount=amount,timestamp=timestamp)

def generate_mule_ring(graph,num_mule_accounts,start_id):
    mule_ids = list(range(start_id,start_id+num_mule_accounts))
    for account_id in mule_ids:
        graph.add_node(account_id,is_mule=True)
    return mule_ids

def add_layering_transactions(graph, mule_ids):
    for i in range(len(mule_ids) - 1):
        source = mule_ids[i]
        target = mule_ids[i + 1]

        base_timestamp = random.randint(0, 8000)

        for hop in range(MULE_TRANSACTIONS_PER_HOP):
            amount = round(random.uniform(4000, 9000), 2)
            timestamp = base_timestamp + hop * 5  # tight time gaps, simulating rapid pass-through

            graph.add_edge(source, target, amount=amount, timestamp=timestamp)


def build_graph():
    graph = nx.DiGraph()
    generate_normal_accounts(graph, NUM_NORMAL_ACCOUNTS)
    add_normal_transactions(graph, NUM_NORMAL_ACCOUNTS)
    mule_ids = generate_mule_ring(graph, NUM_MULE_ACCOUNTS_LAYERING, start_id=NUM_NORMAL_ACCOUNTS)
    add_layering_transactions(graph, mule_ids)
    return graph

if __name__ == "__main__":
    g = build_graph()
    print(f"Total nodes: {g.number_of_nodes()}, total edges: {g.number_of_edges()}")
    mule_nodes = [n for n, d in g.nodes(data=True) if d["is_mule"]]
    print(f"Mule accounts: {mule_nodes}")
    print(f"Sample mule edge: {list(g.edges(mule_nodes[0], data=True))[:2]}")