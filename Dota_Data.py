import asyncio
import json

import aiohttp
import requests
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
get_matches_url = "https://api.opendota.com/api/publicMatches"
get_match_info_url = " https://api.opendota.com/api/matches/{match_id} "

player_id_key = "account_id"


def get_dota_match():
    get_json_from_url(get_matches_url)

def append_key(url):
    return url + "?api_key="+"843e9b63-06c0-4830-b66e-281d044d4182"


def get_json_from_url(url):
    response = requests.get(url)
    print(response)
    response.raise_for_status()
    json_data = response.json()
    return json_data

def write_numbers_to_file(list, file_name):
    with open(file_name, 'w') as file:
        file.write('\n'.join(str(number) for number in list))

def read_numbers_from_file(file_name):
    with open(file_name, 'r') as file:
        return [line.strip() for line in file.readlines()]

def write_2d_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for row in array:
            row_string = ','.join([str(x) for x in row])
            f.write(f'{row_string}\n')


def read_2d_array_from_file(filename):
    array = []
    with open(filename, 'r') as f:
        for line in f:
            row = [int(x) for x in line.strip().split(',')]
            array.append(row)
    return array


def get_match_info(match_id):
    url = "https://api.opendota.com/api/matches/"+str(match_id)
    url = append_key(url)
    return get_json_from_url(url)

def get_match_data(match_id):
    match_json = get_match_info(match_id)
    return match_json

def get_player_info(match_json):
    try:
        radient_win = match_json.get("radiant_win")
        #print("radient_win: ", radient_win, ", ", radient_win.__class__)
        players = match_json.get("players")
        player_ids = []
        id_count = 0
        player_pick = []
        for i in range(10):
            #print("player: ")
            player_json = players[i]

            hero_id = player_json.get("hero_id")
            #print(hero_id)
            player_pick.append(hero_id)
        if radient_win == True:
            player_pick.append(1)
        else:
            player_pick.append(0)
    except:
        return None

    return player_pick


def process_match_info():
    json_data = get_json_from_url(get_matches_url)
    print(json.dumps(json_data, indent=2))
    id_list = []
    for match in json_data:
        id_list.append(match.get("match_id"))
    print(id_list)
    write_numbers_to_file(id_list, "match_ids")
    read_list = read_numbers_from_file("match_ids")
    print("read list: ", read_list)

def get_match_ids(min):
    url = append_key(get_matches_url) + "?less_than_match_id=" + str(min)
    json_data = get_json_from_url(url)
    id_list = []
    for match in json_data:
        id_list.append(match.get("match_id"))
    return id_list

def load_batch_of_matches():
    match_ids = read_numbers_from_file("test_dup.txt")
    print("ids: ", len(match_ids))
    remove_duplicates_1d(match_ids)
    print("remove dup: ", match_ids.__len__())
    base_id = 7116572413
    looping_id_sets = True
    data = []
    print("starting loop")


    for i in range(match_ids.__len__()):
        match_json = get_match_data(match_ids[i])

        # print(match_json)
        picks = get_player_info(match_json)
        data.append(picks)
        # print("read game: \n", picks)
        if i%10 == 0:
            print(i)

        if (data.__len__() >= 500):
            break
    print(data.__len__())

    write_2d_array_to_file(data, "data_2000_2.txt")

    data_100_read = read_2d_array_from_file("data_100.txt")
    print("data_100_read: ", data_100_read)

async def fetch(url, session):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def main():
    urls = [
        'https://httpbin.org/get',
        'https://api.github.com',
        'https://www.example.com',
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(url, session) for url in urls]
        responses = await asyncio.gather(*tasks)

        for url, response_text in zip(urls, responses):
            print(f'Response from {url}:\n{response_text}\n')

async def load_batch_of_matches_brute():
    base_id = 7115882413
    looping_id_sets = True
    data = []
    print("starting loop")
    count = 0
    i = 0
    batch = 5

    while(True):
        batch_urls = []
        if count > 2000:
            break
        for batch_index in range(batch):

            url = append_key("https://api.opendota.com/api/matches/"+str(i+base_id))
            batch_urls.append(url)
            i+=1

        async with aiohttp.ClientSession() as session:
            tasks = [fetch(url, session) for url in batch_urls]
            responses = await asyncio.gather(*tasks)

            for url, response_text in zip(batch_urls, responses):
                print(response_text)
                match_json = None
                try:
                    match_json = json.loads(response_text)
                except:
                    print("404")
                if match_json is None:
                    continue
                picks = get_player_info(match_json)
                if (picks is None):
                    continue
                count += 1
                data.append(picks)
                print(count)
    print(data.__len__())

    write_2d_array_to_file(data, "data_2000_3.txt")

    #data_100_read = read_2d_array_from_file("data_100.txt")
    #print("data_100_read: ", data_100_read)

def get_lists():
    raw_data_1: list = read_2d_array_from_file("data_2000_1.txt")
    raw_data_2: list = read_2d_array_from_file("data_2000_2.txt")
    raw_data_3: list = read_2d_array_from_file("data_2000_3.txt")
    raw_data = []
    for row in raw_data_1:
        raw_data.append(row)
    for row in raw_data_2:
        raw_data.append(row)
    for row in raw_data_3:
        raw_data.append(row)
    raw_data = remove_duplicates_2d(raw_data)
    print("unique data points: ", raw_data.__len__())
    features_list = []
    label_list = []
    for i in range(len(raw_data)):
        row = raw_data[i]
        features_raw = row[0:-1]
        features = []
        for hero in features_raw:
            hero_one_hot = [0] * 140
            #print(hero)
            hero_one_hot[hero] = 1
            features.append(hero_one_hot)
        features_list.append(features)
        label = row[-1]
        if (label == 1):
            label_list.append([1, 0])
        else:
            label_list.append([0, 1])
    return features_list, label_list

def get_arg_max_lists():
    raw_data_1: list = read_2d_array_from_file("data_2000_1.txt")
    raw_data_2: list = read_2d_array_from_file("data_2000_2.txt")
    raw_data_3: list = read_2d_array_from_file("data_2000_3.txt")
    raw_data = []
    for row in raw_data_1:
        raw_data.append(row)
    for row in raw_data_2:
        raw_data.append(row)
    for row in raw_data_3:
        raw_data.append(row)

    features_list = []
    label_list = []
    for i in range(len(raw_data)):
        row = raw_data[i]
        features_raw = row[0:-1]
        features_list.append(features_raw)
        label = row[-1]
        label_list.append(label)
    return features_list, label_list

def load_data_as_one_hot():
    features_list, label_list = get_lists()
    #print("feature_list: ", features_list)
    #print("label_list", label_list)

    #print(features_list[0])
    #print(label_list[0])

    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    labels_tensor = torch.tensor(label_list, dtype=torch.float32)

    print("Features tensor shape:", features_tensor.shape)
    print("Labels tensor shape:", labels_tensor.shape)
    return create_data_loaders(features_tensor, labels_tensor)
    #return {"feature": features_tensor, "label": labels_tensor}

def remove_duplicates_2d(lst):
    return [list(item) for item in set(tuple(sublist) for sublist in lst)]

def remove_duplicates_1d(lst):
    return list(set(lst))

def create_data_loaders(feature_tensor, label_tensor, batch_size=64, test_size=0.2, random_state=42):
    # Split the dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        feature_tensor, label_tensor, test_size=test_size, random_state=random_state
    )
    print("creating loaders")
    print("feature_tensor: ", feature_tensor.shape)
    print("label_tensor: ", label_tensor.shape)
    print("train_features: ", train_features.shape)
    print("test_features: ", test_features.shape)
    # Create TensorDataset objects for train and test sets
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "test": test_loader}

def write_match_id_file():
    count = 0
    ids = []
    while count < 1000:
        min = get_min(ids)
        print("new min: ", min)
        temp_ids = get_match_ids(min)
        for id in temp_ids:
            ids.append(id)
        count += 100
    remove_duplicates_1d(ids)
    print(ids.__len__())

    write_numbers_to_file(ids, "test_dup.txt")

def get_min(ids):
    if len(ids) < 1:
        return 7116574001
    min = ids[0]
    for num in ids:
        if num < min:
            min = num
    return min

def get_win_rates():
    hero_wins = {}
    hero_loss = {}
    for i in range(140):
        hero_wins.update({i: 0})
        hero_loss.update({i: 0})

    features, label = get_arg_max_lists()

    for row in range(len(features)):
        heroes = features[row]
        print("heroes: ", heroes)

        win = label[row]
        print(win)
        for radiant in range(5):
            hero = heroes[radiant]
            if(win == 1):
                wins = hero_wins.get(hero)
                wins += 1
                hero_wins.update({hero: wins})
            else:
                loss = hero_loss.get(hero)
                loss += 1
                hero_loss.update({hero: loss})
        for dire in range(5,9):
            hero = heroes[dire]
            if(win == 0):
                wins = hero_wins.get(hero)
                wins += 1
                hero_wins.update({hero: wins})
            else:
                loss = hero_loss.get(hero)
                loss += 1
                hero_loss.update({hero: loss})
    print(hero_wins)
    print(hero_loss)

    hero_winrate = []
    for i in range(140):
        if hero_wins.get(i) + hero_loss.get(i) == 0:
            hero_winrate.append(0)
            continue
        hero_winrate.append(hero_wins.get(i) / (hero_wins.get(i) + hero_loss.get(i)))

    print("winrate: ", hero_winrate)





if __name__ == "__main__":
    get_win_rates()
    #load_data_as_one_hot()
    #write_match_id_file()
    #load_batch_of_matches()
    #asyncio.run(load_batch_of_matches_brute())











