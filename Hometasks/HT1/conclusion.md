
1. Был проведён разведочный анализ данных. 
В результате которого были заполнены пропуски в данных.
Затем мы сделали визуализацию, чтобы увидеть как распределены значения признаков.
Мы нашли признаки, которые сильнее всего влияют на целевую переменную.
2. Затем мы обучали линейные модели с перебором гиперпараметров, для поиска наилучшей модели для нашего случая.
3. Добавили категориальные признаки, что позволило улучшить качество модели ~ 3%.
4. Реализовали кастомную метрику качества модели.
5. Реализовали сервис, позволяющий делать предсказания по данным.
Вывод: Если достигаем потолка по качеству линейной модели и перебор гиперпараметров не помогает.
Нужно создать новые фичи, возможно это позволит улучшить качество модели.