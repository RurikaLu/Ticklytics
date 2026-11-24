import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/dataset.csv")

print("=" * 50)
print("Проверка данных")
print("=" * 50)
print(f"Строк: {len(df)}, Колонок: {len(df.columns)}")
print(f"Пропусков: {df.isnull().sum().sum()}")
print(f"Дубликатов: {df.duplicated().sum()}")
print(f"Диапазон PCR: [{df['PCR'].min():.2f}, {df['PCR'].max():.2f}]")
print(f"Диапазон DSE: [{df['DSE'].min():.2f}, {df['DSE'].max():.2f}]")
print(f"Случаев TC > TP: {(df['TC'] > df['TP']).sum()}")
print(f"DOR = 0: {(df['DOR'] == 0).sum()}")

# Проверка выбросов по IQR для PCR
q1, q3 = df['PCR'].quantile([0.25, 0.75])
iqr = q3 - q1
outliers_pcr = df[(df['PCR'] < q1 - 1.5*iqr) | (df['PCR'] > q3 + 1.5*iqr)]
print(f"Выбросов по PCR (IQR): {len(outliers_pcr)} ({len(outliers_pcr)/len(df)*100:.1f}%)")
# ========== 1. Базовые метрики по этапам ==========

print("=" * 60)
print("Анализ по этапам исследования (grade = 21, 22, 23)")
print("=" * 60)

# Группировка по этапу
stage_stats = df.groupby('grade').agg({
    'PCR': ['mean', 'median', 'std', 'count'],
    'TP': 'mean',
    'TC': 'mean',
    'DIL': 'mean',
    'DET': 'mean',
    'DSE': 'mean',
    'TME': 'mean',
    'DOR': 'mean',
    'ccc': 'mean'
}).round(2)

print("\nСредние метрики по этапам:")
print(stage_stats)

print("\n" + "=" * 60)
print("Ключевые цифры по этапам")
print("=" * 60)

for stage in sorted(df['grade'].unique()):
    subset = df[df['grade'] == stage]
    print(f"\n--- Этап {int(stage)} (n={len(subset)}) ---")
    print(f"  PCR среднее: {subset['PCR'].mean():.2f}, медиана: {subset['PCR'].median():.2f}")
    print(f"  Доля с PCR < 0.8: {(subset['PCR'] < 0.8).mean()*100:.1f}%")
    print(f"  Доля с PCR < 0.5: {(subset['PCR'] < 0.5).mean()*100:.1f}%")
    print(f"  TP (план) vs TC (факт): {subset['TP'].mean():.1f} vs {subset['TC'].mean():.1f}")
    print(f"  DIL (учёба): {subset['DIL'].mean():.1f} мин/день")
    print(f"  DET (внеучебка): {subset['DET'].mean():.1f} мин/день")
    print(f"  DSE (самооценка): {subset['DSE'].mean():.2f}")
    print(f"  TME (опыт): {subset['TME'].mean():.2f}")
    print(f"  DOR (дней записи): {subset['DOR'].mean():.1f}")

# Графики

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

stages = sorted(df['grade'].unique())
colors = ['#4C72B0', '#DD8452', '#55A868']

# 1 Выполнение плана
axes[0, 0].boxplot([df[df['grade'] == s]['PCR'].dropna() for s in stages],
                    tick_labels=[f'Этап {int(s)}' for s in stages],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
axes[0, 0].axhline(1.0, color='green', linestyle=':', linewidth=2, label='Идеал: 1.0')
axes[0, 0].set_ylabel('Коэффициент выполнения плана')
axes[0, 0].set_title('Выполнение плана по этапам исследования')
axes[0, 0].legend()

# 2 Сколько запланировано и выполнено
x = np.arange(len(stages))
width = 0.35
tp_means = [df[df['grade'] == s]['TP'].mean() for s in stages]
tc_means = [df[df['grade'] == s]['TC'].mean() for s in stages]
axes[0, 1].bar(x - width/2, tp_means, width, label='Запланировано', color='#4C72B0')
axes[0, 1].bar(x + width/2, tc_means, width, label='Выполнено', color='#DD8452')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'Этап {int(s)}' for s in stages])
axes[0, 1].set_ylabel('Среднее количество задач')
axes[0, 1].set_title('Количество выполненных и запланированных задач')
axes[0, 1].legend()

# 3 Распределение времени на учёбу и внеучебку
dil_means = [df[df['grade'] == s]['DIL'].mean() for s in stages]
det_means = [df[df['grade'] == s]['DET'].mean() for s in stages]
axes[0, 2].plot(stages, dil_means, 'o-', label='Учёба', color='#4C72B0', linewidth=2)
axes[0, 2].plot(stages, det_means, 's-', label='Внеучебная деятельность', color='#C44E52', linewidth=2)
axes[0, 2].set_xticks(stages)
axes[0, 2].set_xticklabels([f'Этап {int(s)}' for s in stages])
axes[0, 2].set_ylabel('Минут в день')
axes[0, 2].set_title('Распределение времени')
axes[0, 2].legend()

# 4 Самооценка
dse_means = [df[df['grade'] == s]['DSE'].mean() for s in stages]
axes[1, 0].bar(stages, dse_means, color=colors, edgecolor='black')
axes[1, 0].set_xticks(stages)
axes[1, 0].set_xticklabels([f'Этап {int(s)}' for s in stages])
axes[1, 0].set_ylabel('Средняя самооценка студентов')
axes[1, 0].set_title('Изменение самооценки')
axes[1, 0].set_ylim(0, 5)

# 5 Про опыт управления временем
median_tme = df['TME'].median()
novice_count = len(df[df['TME'] <= median_tme])
exp_count = len(df[df['TME'] > median_tme])
labels = [f'Ниже среднего\n({novice_count} чел.)', f'Выше среднего\n({exp_count} чел.)']
sizes = [novice_count, exp_count]
pie_colors = ['#C44E52', '#55A868']
explode = (0.1, 0)
axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=pie_colors,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}, shadow=True)
axes[1, 1].set_title('Уровень опыта управления временем')

# 6 "Как стабильно" вели записи о выполнении задач
dor_means = [df[df['grade'] == s]['DOR'].mean() for s in stages]
axes[1, 2].bar(stages, dor_means, color=colors, edgecolor='black')
axes[1, 2].set_xticks(stages)
axes[1, 2].set_xticklabels([f'Этап {int(s)}' for s in stages])
axes[1, 2].set_ylabel('Среднее кол-во дней ведения записей')
axes[1, 2].set_title('Регулярность ведения записей')

plt.tight_layout()
plt.savefig('analysis_by_stages.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафик сохранён в analysis_by_stages.png")
