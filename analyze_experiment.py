# Анализ experiment_data.csv. Запуск: python analyze_experiment.py
import pandas as pd
from scipy.stats import chisquare, mannwhitneyu, fisher_exact

ALPHA = 0.05
GROUPS = ['control', 'test']
MY_SEED = 648345

# Скрипт читает CSV из текущей папки.
df = pd.read_csv('experiment_data.csv', parse_dates=['date'])

print('Размер данных:', df.shape)
print(df.head().to_string(index=False))
print('\nПропуски по столбцам:')
print(df.isna().sum().to_string())
print('\nПовторяющихся user_id:', df['user_id'].duplicated().sum())

assert not df.isna().any().any(), 'В данных есть пропуски'
assert df['user_id'].is_unique, 'Есть повторяющиеся пользователи'
assert set(df['variant'].unique()) == set(GROUPS), 'Неверные группы'
assert df['converted'].isin([0, 1]).all(), 'Неверные значения converted'
assert df['error_flag'].isin([0, 1]).all(), 'Неверные значения error_flag'
assert (df['pre_activity'] >= 0).all(), 'Отрицательная активность'

print('\nБазовые проверки пройдены.')
print('Даты:', df['date'].min().date(), '—', df['date'].max().date())

# Среднее бинарного признака равно доле единиц.
summary = df.groupby('variant').agg(
    users=('user_id', 'size'),
    conversions=('converted', 'sum'),
    conversion_rate=('converted', 'mean'),
    errors=('error_flag', 'sum'),
    error_rate=('error_flag', 'mean'),
    activity_mean=('pre_activity', 'mean'),
    activity_median=('pre_activity', 'median')
).reindex(GROUPS)

summary['errors'] = summary['errors'].astype(int)
summary['group_pct'] = 100 * summary['users'] / len(df)
summary['conversion_pct'] = 100 * summary['conversion_rate']
summary['error_pct'] = 100 * summary['error_rate']

print(summary[[
    'users', 'group_pct', 'conversions', 'conversion_pct',
    'errors', 'error_pct', 'activity_mean', 'activity_median'
]].round(3).to_string())

control = df[df['variant'] == 'control']
test = df[df['variant'] == 'test']
warnings = []

observed = [len(control), len(test)]
expected = [len(df) / 2, len(df) / 2]
chi2_stat, p_srm = chisquare(observed, f_exp=expected)

print('Наблюдаемые численности:', observed)
print('Ожидаемые численности:', expected)
print(f'Хи-квадрат = {chi2_stat:.3f}; p-value = {p_srm:.6g}')

if p_srm < ALPHA:
    warnings.append('SRM обнаружен')
    print('SRM обнаружен')
else:
    print('Статистически значимый SRM не обнаружен')

# Распределение скошено, а целые значения часто повторяются.
# Асимптотический вариант U-теста учитывает совпадающие ранги.
u_stat, p_activity = mannwhitneyu(
    control['pre_activity'], test['pre_activity'],
    alternative='two-sided', method='asymptotic'
)

print(f"Среднее control: {control['pre_activity'].mean():.3f}")
print(f"Среднее test: {test['pre_activity'].mean():.3f}")
print(f'U = {u_stat:.1f}; p-value = {p_activity:.6g}')

if p_activity < ALPHA:
    warnings.append('Дисбаланс по pre_activity — возможен confounder')
    print('Дисбаланс по pre_activity — возможен confounder')
else:
    print('Значимый дисбаланс pre_activity не обнаружен')

errors_control = int(control['error_flag'].sum())
errors_test = int(test['error_flag'].sum())

# Строки: control, test. Столбцы: ошибка, нет ошибки.
error_table = [
    [errors_control, len(control) - errors_control],
    [errors_test, len(test) - errors_test]
]
_, p_error = fisher_exact(error_table, alternative='two-sided')

error_control = control['error_flag'].mean()
error_test = test['error_flag'].mean()
print('Таблица частот:', error_table)
print(f'Частота ошибок control: {error_control:.3%}')
print(f'Частота ошибок test: {error_test:.3%}')
print(f'Разница: {100 * (error_test - error_control):+.3f} п.п.')
if error_control > 0:
    print(f'Отношение частот test / control: {error_test / error_control:.2f}')
print(f'p-value = {p_error:.6g}')

if p_error < ALPHA:
    warnings.append('Дисбаланс guardrail: частота ошибок различается')
    if error_test > error_control:
        print('Guardrail ухудшился: в тесте значимо больше ошибок')
    else:
        print('Обнаружено значимое различие частоты ошибок')
else:
    print('Значимое различие частоты ошибок не обнаружено')

checks = pd.DataFrame({
    'Проверка': ['SRM', 'Баланс pre_activity', 'Guardrail'],
    'p_value': [p_srm, p_activity, p_error],
    'Предупреждение': [
        p_srm < ALPHA, p_activity < ALPHA, p_error < ALPHA
    ]
})
print(checks.to_string(index=False))

if warnings:
    print('\nЭксперимент не валиден.')
    for warning in warnings:
        print('-', warning)
    print('Остановить тест, исправить ошибки, перезапустить')
    print('Проверка значимости конверсии не выполняется.')
else:
    print('\nПредупреждений нет. По проверкам задания эксперимент валиден.')
    conversions_control = int(control['converted'].sum())
    conversions_test = int(test['converted'].sum())
    conversion_table = [
        [conversions_control, len(control) - conversions_control],
        [conversions_test, len(test) - conversions_test]
    ]
    _, p_conversion = fisher_exact(conversion_table, alternative='two-sided')
    print(f'p-value конверсии = {p_conversion:.6g}')
    if p_conversion < ALPHA:
        print('Различие конверсий статистически значимо.')
    else:
        print('Статистически значимое различие конверсий не обнаружено.')

parts = []

for activity, group in df.groupby('pre_activity'):
    group_control = group[group['variant'] == 'control']
    group_test = group[group['variant'] == 'test']
    count = min(len(group_control), len(group_test))

    if count > 0:
        parts.append(group_control.sample(n=count, random_state=MY_SEED))
        parts.append(group_test.sample(n=count, random_state=MY_SEED))

balanced = pd.concat(parts, ignore_index=True)
balanced_control = balanced[balanced['variant'] == 'control']
balanced_test = balanced[balanced['variant'] == 'test']

print('До отбора:', len(df))
print('После отбора:', len(balanced))
print(f'Исключено: {len(df) - len(balanced)} '
      f'({100 * (1 - len(balanced) / len(df)):.2f}%)')

# Проверяем именно полное совпадение частот значений активности.
activity_counts = pd.crosstab(balanced['pre_activity'], balanced['variant'])
print('Распределения активности совпадают:',
      activity_counts['control'].equals(activity_counts['test']))

balanced_summary = balanced.groupby('variant').agg(
    users=('user_id', 'size'),
    activity_mean=('pre_activity', 'mean'),
    conversion_rate=('converted', 'mean'),
    error_rate=('error_flag', 'mean')
).reindex(GROUPS)

comparison = pd.concat({
    'До отбора': summary[[
        'users', 'activity_mean', 'conversion_rate', 'error_rate'
    ]],
    'После отбора': balanced_summary
})
comparison['conversion_pct'] = 100 * comparison['conversion_rate']
comparison['error_pct'] = 100 * comparison['error_rate']
print(comparison[[
    'users', 'activity_mean', 'conversion_pct', 'error_pct'
]].round(3).to_string())

# Эти p-value показывают результат отбора, но не подтверждают рандомизацию.
_, p_srm_balanced = chisquare(
    [len(balanced_control), len(balanced_test)],
    f_exp=[len(balanced) / 2, len(balanced) / 2]
)
_, p_activity_balanced = mannwhitneyu(
    balanced_control['pre_activity'], balanced_test['pre_activity'],
    alternative='two-sided', method='asymptotic'
)

errors_control_balanced = int(balanced_control['error_flag'].sum())
errors_test_balanced = int(balanced_test['error_flag'].sum())
error_table_balanced = [
    [errors_control_balanced, len(balanced_control) - errors_control_balanced],
    [errors_test_balanced, len(balanced_test) - errors_test_balanced]
]
_, p_error_balanced = fisher_exact(error_table_balanced, alternative='two-sided')

print(f'SRM после отбора: p-value = {p_srm_balanced:.6g}')
print(f'pre_activity после отбора: p-value = {p_activity_balanced:.6g}')
print('Ошибки после отбора:', error_table_balanced)
print(f'Guardrail после отбора: p-value = {p_error_balanced:.6g}')

conversion_diff_before = test['converted'].mean() - control['converted'].mean()
conversion_diff_after = (
    balanced_test['converted'].mean() - balanced_control['converted'].mean()
)
print(f'Разница конверсий до отбора: {100 * conversion_diff_before:+.3f} п.п.')
print(f'Разница конверсий после отбора: {100 * conversion_diff_after:+.3f} п.п.')
print('Конверсия после отбора показана только описательно.')

if p_error_balanced < ALPHA:
    print('После балансировки различие guardrail сохранилось.')
print('Отбор не делает исходный эксперимент валидным. Нужен перезапуск.')