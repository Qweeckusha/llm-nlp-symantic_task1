import pandas as pd
import sqlite3
import re

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


def load_texts(file_path):
    """
    Загружает текстовые данные из БД

    args:
        file_path: путь к БД с новостными статьями

    output:
        df: DataFrame с одним столбцом текстов статей
    """
    with sqlite3.connect(file_path) as conn:
        query = """
                SELECT description as text \
                FROM articles \
                WHERE description IS NOT NULL \
                """

        df = pd.read_sql(query, conn)
    return df



def clean_text(text):
    """
    Очищает текст от аномалий парсинга по возможности (не от всех)

    :param text: текст, который пройдёт обработку

    :returns text.strip(): Обработанный текст

    """
    if not isinstance(text, str):
        return ""

    # 1. Замена спецсимволов переноса на пробелы
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 2. Разделение стыков Латиница/Кириллица
    text = re.sub(r'([a-zA-Z])([а-яА-Я])', r'\1 \2', text)
    text = re.sub(r'([а-яА-Я])([a-zA-Z])', r'\1 \2', text)

    # 3. Обработка ЗНАКОВ ПРЕПИНАНИЯ
    punctuation = ',.:;!?'

    # 3а. Убираем пробелы ПЕРЕД этими знаками ("слово ,") -> "слово,"
    pattern_before = r'\s+([' + re.escape(punctuation) + r'])'
    text = re.sub(pattern_before, r'\1', text)

    # 3б. Добавляем пробел ПОСЛЕ этих знаков, если там сразу буква ("слово,слово") -> "слово, слово"
    pattern_after = r'([' + re.escape(punctuation) + r'])([А-Яа-яA-Za-z])'
    text = re.sub(pattern_after, r'\1 \2', text)

    # 4. Обработка СКОБОК ()
    # 4а. Открывающая скобка: убираем пробел ВНУТРИ после неё "( слово)" -> "(слово)"
    text = re.sub(r'\(\s+', r'(', text)
    # 4б. Открывающая скобка: добавляем пробел ПЕРЕД ней, если там буква "слово(" -> "слово ("
    text = re.sub(r'([А-Яа-яA-Za-z0-9])\(', r'\1 (', text)

    # 4в. Закрывающая скобка: убираем пробел ВНУТРИ перед ней "(слово )" -> "(слово)"
    text = re.sub(r'\s+\)', r')', text)
    # 4г. Закрывающая скобка: добавляем пробел ПОСЛЕ неё, если там буква ")слово" -> ") слово"
    text = re.sub(r'\)([А-Яа-яA-Za-z0-9])', r') \1', text)

    # 5. все множественные пробелы превращаем в один
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_pairs(doc):
    """

    Генератор, который выдает все пары (подлежащее, сказуемое) в предложении

    :param doc: объект/текст (например запись из датафрейма) для обработки spacy и лемматизации

    :return
        found_pairs: список всех пар кортежами (подлежащее, сказуемое) с леммами для doc

    """

    found_pairs = []
    for sent in doc.sents:
        predicates = []

        for token in sent:
            # Проверка глаголов и вспомогательных глаголов
            if token.pos_ in ['VERB', 'AUX']:
                # Поиска среди найденных сказуемого
                if token.dep_ == "ROOT":
                    predicates.append(token)
                elif token.dep_ == "conj":
                    predicates.append(token)

        # Для каждого сказуемого ищем его подлежащее
        for pred in predicates:
            subject = None
            # Ищем среди детей сказуемого
            for child in pred.children:
                if child.dep_ == "nsubj":
                    subject = child
                    break

            # Если пара найдена, сохраняем леммы в нижнем регистре
            if subject and subject.pos_ in ["NOUN", "PROPN", "PRON"]: # Подлежащее обычно существительное, имя собственное или местоимение
                found_pairs.append((subject.lemma_.lower(), pred.lemma_.lower()))

    return found_pairs


def multithread_spacy_proccesing (df, nlp):

    """

    :param df: DataFrame состоящий из одного столбца
    :type df: pandas.DataFrame


    :param nlp: модель spacy, предварительно её нужно загрузить python -m spacy download ru_core_news_{sm/md/lg}
    :type spacy.lang.{lang}.{Language}

    :return
        results: Список списков (по текстам статей) кортежей (пары подлежащее, сказуемое)
    :type [[(), ()], [(), ()]]

    """

    texts = df.fillna('').tolist()

    results = []

    #Основная функция многопоточности
    docs_stream = nlp.pipe(texts, batch_size=50, n_process=9)

    for doc in tqdm(docs_stream, total=len(texts), desc="Обработка"):
        pairs = extract_pairs(doc)
        results.append(pairs)

    return results


def vizualization(counter, top_n=20):

    """

    :param counter: счётчик кортежей/пар подлежащее - сказуемое

    :param top_n: количество в топе самых частых

    :return: pyplots визуализация на графиках barplot (Гистограмма) и heatmap (Корреляционная матрица)

    """

    top = counter.most_common(top_n) # кортеж и частота

    df_viz = pd.DataFrame(top, columns=['Pair', 'Count'])

    df_viz['Subject'] = df_viz['Pair'].apply(lambda x: x[0])
    df_viz['Verb'] = df_viz['Pair'].apply(lambda x: x[1])
    df_viz['Label'] = df_viz['Subject'] + ' - ' + df_viz['Verb']

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams['figure.figsize'] = (12, 8)

    plt.figure(figsize=(10, 8))
    # Используем barplot, сортируем по Count автоматически
    sns.barplot(data=df_viz, x='Count', y='Label', palette='YlOrRd')
    plt.title(f'Топ-{top_n} наиболее частых пар (Подлежащее — Сказуемое)', fontsize=16, weight='bold')
    plt.xlabel('Частота встречаемости', fontsize=12)
    plt.ylabel('Грамматическая пара', fontsize=12)
    plt.tight_layout()
    plt.show()

    top_subjects = [p[0][0] for p in counter.most_common(100)]
    top_verbs = [p[0][1] for p in counter.most_common(100)]

    unique_subjects = list(dict.fromkeys(top_subjects))[:15]
    unique_verbs = list(dict.fromkeys(top_verbs))[:15]

    matrix_data = []
    for subj in unique_subjects:
        row = []
        for verb in unique_verbs:
            count = counter.get((subj, verb), 0)
            row.append(count)
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data, index=unique_subjects, columns=unique_verbs)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Частота'})
    plt.title('Матрица совместимости: Топ-10 Подлежащих vs Топ-10 Сказуемых', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()