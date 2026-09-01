import random
import numpy as np
import networkx as nx


# Config

RANDOM_SEED = 42

NUM_NORMAL_ACCOUNTS = 300
NORMAL_TRANSACTIONS_PER_ACCOUNT = 10
NORMAL_AMOUNT_RANGE = (100, 5000)

NUM_LAYERING_ACCOUNTS = 8
LAYERING_TRANSACTIONS_PER_HOP = 12
LAYERING_AMOUNT_RANGE = (4000, 9000)
LAYERING_HOP_GAP_SECONDS = 5

NUM_FUNNEL_COLLECTORS = 6
NUM_FUNNEL_FEEDERS = 15
FUNNEL_AMOUNT_RANGE = (3000, 8000)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



# Normal accounts

def add_normal_accounts(graph):
    for account_id in range(NUM_NORMAL_ACCOUNTS):
        graph.add_node(account_id, is_mule=False)


def add_normal_transactions(graph):
    for account_id in range(NUM_NORMAL_ACCOUNTS):
        num_transactions = random.randint(1, NORMAL_TRANSACTIONS_PER_ACCOUNT)
        for _ in range(num_transactions):
            target = _random_other_account(account_id, NUM_NORMAL_ACCOUNTS)
            amount = round(random.uniform(*NORMAL_AMOUNT_RANGE), 2)
            timestamp = random.randint(0, 10000)
            graph.add_edge(account_id, target, amount=amount, timestamp=timestamp)


def _random_other_account(exclude_id, pool_size):
    target = random.randint(0, pool_size - 1)
    while target == exclude_id:
        target = random.randint(0, pool_size - 1)
    return target


# Layering chain

def add_layering_ring(graph, start_id):
    mule_ids = list(range(start_id, start_id + NUM_LAYERING_ACCOUNTS))
    for account_id in mule_ids:
        graph.add_node(account_id, is_mule=True)

    for i in range(len(mule_ids) - 1):
        source, target = mule_ids[i], mule_ids[i + 1]
        base_timestamp = random.randint(0, 8000)
        for hop in range(LAYERING_TRANSACTIONS_PER_HOP):
            amount = round(random.uniform(*LAYERING_AMOUNT_RANGE), 2)
            timestamp = base_timestamp + hop * LAYERING_HOP_GAP_SECONDS
            graph.add_edge(source, target, amount=amount, timestamp=timestamp)

    return mule_ids


# Funnel pattern


def add_funnel_ring(graph, start_id):
    collector_ids = list(range(start_id, start_id + NUM_FUNNEL_COLLECTORS))
    for account_id in collector_ids:
        graph.add_node(account_id, is_mule=True)

    feeder_accounts = random.sample(range(NUM_NORMAL_ACCOUNTS), NUM_FUNNEL_FEEDERS)
    for feeder in feeder_accounts:
        collector = random.choice(collector_ids)
        for _ in range(random.randint(2, 5)):
            amount = round(random.uniform(*FUNNEL_AMOUNT_RANGE), 2)
            timestamp = random.randint(0, 10000)
            graph.add_edge(feeder, collector, amount=amount, timestamp=timestamp)

    return collector_ids



# Assemble


def build_graph():
    graph = nx.DiGraph()

    add_normal_accounts(graph)
    add_normal_transactions(graph)

    layering_ids = add_layering_ring(graph, start_id=NUM_NORMAL_ACCOUNTS)
    funnel_ids = add_funnel_ring(graph, start_id=NUM_NORMAL_ACCOUNTS + NUM_LAYERING_ACCOUNTS)

    mule_ids = layering_ids + funnel_ids
    return graph, mule_ids


if __name__ == "__main__":
    g, mule_ids = build_graph()
    print(f"Total nodes: {g.number_of_nodes()}, total edges: {g.number_of_edges()}")
    print(f"Mule accounts ({len(mule_ids)}): {mule_ids}")
    print(f"Sample layering edge: {list(g.edges(mule_ids[0], data=True))[:1]}")
    print(f"Sample funnel edge: {list(g.edges(mule_ids[-1], data=True))[:1]}")