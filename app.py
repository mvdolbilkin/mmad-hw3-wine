import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Тип вина',
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_batch(model, batch, features, threshold):
    missing = [column for column in features if column not in batch.columns]
    if missing:
        raise ValueError('В CSV не хватает столбцов: ' + ', '.join(missing))
    if batch.empty:
        raise ValueError('В CSV нет строк с данными.')
    try:
        X = batch.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('Все признаки должны быть числами. Проверьте CSV.') from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать 0 (белое) или 1 (красное).')
    result = batch.copy()
    result['probability_good'] = model.predict_proba(X)[:, 1]
    result['prediction'] = (result['probability_good'] >= threshold).astype(int)
    return result


def main():
    st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')
    st.title('🍷 Винная подборка')
    st.caption('Хорошее вино — экспертная оценка не ниже 7. Модель использует лабораторные признаки.')

    missing_files = [name for name in ['model.joblib', 'test.csv', 'config.json']
                     if not (APP_DIR / name).is_file()]
    if missing_files:
        st.error('Отсутствуют файлы модели: ' + ', '.join(missing_files)
                 + '. Выполните ячейку сохранения модели в ноутбуке ДЗ-3.')
        st.stop()
    model = load_model()
    test = load_test_data()
    config = load_config()
    features = list(model.feature_names_in_)
    X_test = test.loc[:, features]
    y_test = test['target'].to_numpy()
    proba = test_proba(model, X_test)

    threshold = st.sidebar.slider(
        'Порог отбора', min_value=0.0, max_value=1.0,
        value=float(config['threshold']), step=0.001, format='%.4f', key='threshold')
    st.sidebar.caption('Один порог для оценки вина, метрик и CSV. '
                       'Значение по умолчанию выбрано на валидации.')
    st.sidebar.write(f'В подборку попадает вино с вероятностью ≥ {threshold:.4f}.')
    one, quality, batch_tab, coefficients = st.tabs(
        ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

    with one:
        st.subheader('Лабораторный анализ вина')
        st.caption('Диапазоны и медианы взяты из test.csv; единицы совпадают с исходными данными.')
        with st.form('single_wine'):
            columns = st.columns(3)
            values = {}
            numeric_features = [feature for feature in features if feature != 'is_red']
            for i, feature in enumerate(numeric_features):
                values[feature] = columns[i % 3].number_input(
                    LABELS[feature], min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()), format='%.5f',
                    key=f'feature_{feature}')
            values['is_red'] = int(st.toggle('Красное вино', value=bool(X_test['is_red'].median())))
            st.caption('Переключатель выключен — белое вино; включён — красное.')
            submitted = st.form_submit_button('Оценить вино')
        if submitted:
            row = pd.DataFrame([values], columns=features)
            st.session_state['single_probability'] = float(model.predict_proba(row)[0, 1])
        if 'single_probability' in st.session_state:
            probability = st.session_state['single_probability']
            st.metric('Вероятность хорошего вина', f'{probability:.1%}')
            st.progress(probability)
            if probability >= threshold:
                st.success('Рекомендовать в подборку')
            else:
                st.info('Не включать в подборку при текущем пороге')

    with quality:
        st.subheader('Как порог меняет подборку')
        st.write('Увеличение порога сокращает подборку и обычно повышает долю хороших вин, '
                 'но больше хороших вин остаётся за её пределами. Уменьшение порога расширяет '
                 'подборку и повышает полноту, но может добавить посредственные вина. '
                 'Точность на конкретной выборке меняется не обязательно монотонно.')
        st.caption('Тестовая выборка служит для демонстрации. Подбирать рабочий порог по ней нельзя.')
        pred = (proba >= threshold).astype(int)
        cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision = metrics.precision_score(y_test, pred, zero_division=0)
        recall = metrics.recall_score(y_test, pred, zero_division=0)
        f1 = metrics.f1_score(y_test, pred, zero_division=0)
        cards = st.columns(4)
        cards[0].metric('Precision', f'{precision:.3f}')
        cards[1].metric('Recall', f'{recall:.3f}')
        cards[2].metric('F1', f'{f1:.3f}')
        cards[3].metric('Вин в подборке', f'{pred.sum()} / {len(pred)}')
        fig, ax = plt.subplots(figsize=(5, 3))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set(xlabel='Прогноз', ylabel='Факт')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f'FP = {fp}: посредственные вина в подборке. '
                   f'FN = {fn}: хорошие вина, которые пропущены.')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        metrics.RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name='LR')
        axes[0].plot([0, 1], [0, 1], '--', color='grey')
        axes[0].scatter(fp / (fp + tn), recall, color='red', zorder=5, label=f'Порог {threshold:.4f}')
        metrics.PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name='LR')
        axes[1].axhline(y_test.mean(), ls='--', color='grey', label='Случайная модель')
        axes[1].scatter(recall, precision, color='red', zorder=5, label=f'Порог {threshold:.4f}')
        axes[0].legend(); axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption('Кривые и площади под ними от порога не меняются; меняются отмеченные точки. '
                   'Если положительных прогнозов нет, precision и F1 показаны как 0.')

    with batch_tab:
        st.subheader('Оценить CSV с винами')
        st.write('Загрузите CSV с 12 признаками без target. '
                 'is_red: 1 — красное, 0 — белое. Разделитель — запятая, десятичный знак — точка.')
        st.code(','.join(features), language=None)
        st.download_button('Скачать пример CSV', X_test.head(10).to_csv(index=False).encode('utf-8'),
                           file_name='wine_example.csv', mime='text/csv')
        uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
        if uploaded is not None:
            try:
                batch = pd.read_csv(uploaded)
                result = predict_batch(model, batch, features, threshold)
            except pd.errors.EmptyDataError:
                st.error('CSV-файл пуст. Добавьте заголовки и строки с данными.')
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(str(exc))
            else:
                st.dataframe(result, hide_index=True)
                st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8'),
                                   file_name='wine_predictions.csv', mime='text/csv')

    with coefficients:
        st.subheader('Какие признаки связаны с качеством')
        weights = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([LABELS[name] for name in weights.index], weights,
                color=np.where(weights >= 0, '#2a9d8f', '#e76f51'))
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set(xlabel='Коэффициент после StandardScaler', title='Влияние на логарифм шансов класса 1')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        positive = weights[weights > 0].nlargest(3)
        negative = weights[weights < 0].nsmallest(3)
        st.write('Сильнее всего за хорошее вино: ' + ', '.join(LABELS[x] for x in positive.index) + '.')
        st.write('Сильнее всего против: ' + ', '.join(LABELS[x] for x in negative.index) + '.')
        st.caption('Коэффициент относится к увеличению признака на одно стандартное отклонение '
                   'при остальных фиксированных признаках. Это связь в модели, а не доказательство '
                   'причинного влияния; коррелирующие признаки могут делить влияние между собой.')


if __name__ == '__main__':
    main()
