import os
import pandas as pd
import dill
import json
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    mod = sorted(os.listdir(f'{path}/data/models'))

    # загружаем обученную модель

    with open(f'{path}/data/models/{mod[-1]}', 'rb') as file:
        model = dill.load(file)

    # делаем предсказания для всех объектов в папке data\test
    df_pred = pd.DataFrame(columns=['id', 'predict'])
    files_list = os.listdir(f'{path}/data/test')

    # объединяем предсказания в один Dataframe и сохраняем их в csv-формате в папку data/predictions
    for filename in files_list:
        with open(f'{path}/data/test/{filename}', 'r') as file:
            form = json.load(file)
        data = pd.DataFrame.from_dict([form])
        prediction = model.predict(data)

        dict_pred = {'id': data['id'].values[0], 'predict': prediction[0]}
        df = pd.DataFrame([dict_pred])
        df_pred = pd.concat([df, df_pred], ignore_index=True)

    now = datetime.now().strftime("%Y%m%d%H%M")
    df_pred.to_csv(f'{path}/data/predictions/{now}.csv', index=False)


if __name__ == '__main__':
    predict()
