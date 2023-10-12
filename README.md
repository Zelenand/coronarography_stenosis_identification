# Интеллектуальная система идентификации стенозов коронарных артерий у больных ишемической болезнью сердца



## Основная информация

Разрабатываемая система предназначена для анализа изображений коронарных артерий у больных ишемической болезнью сердца и автоматической идентификации на них стенозов.

Стеноз коронарной артерии - стойкое сужение просвета сосудов, по которым к сердцу поступает кислород и питательные вещества, необходимые для его нормальной работы.
## Участники проекта

- Самарин Никита - мультименеджер/разработчик
- Бойко Антон - разработчик
- Ершов Тимофей - разработчик

## Основные требования/Бэклог

Система должна при подаче изображения коронарной артерии преобразовывать его в подходящую для анализа форму(при необходимости), идентифицировать стенозы и выдавать отредактированное изображение с указанными идентифицированными стенозами. Всё должно происходить с минимальным участием врача/пользователя.

#### Бэклог основных функций(в порядке убывания приоритета)
| Название |
| ------ |
|Импорт изображения с помощью графического интерфейса|        |
|Идентификация стенозов с помощи модели машинного обучения|        |
|Выдача редактированного изображения с указанием стенозов|        |

## Связанные исследования

В процессе подготовки к работе были изучены научные статьи связанные с темой интеллектуального анализа коронарографии. Был составлен google-документ с структурированной краткой информацией из данных статей (выполняемые задачи, используемые методы/датасеты, результаты):
https://docs.google.com/document/d/1sMTnnQiP7YtuHqmUsG6wkOQKimmTDM1ZP52s2G5YF2g/edit?usp=sharing

## Используемые методы

Для решения задачи локализации стеноза на
основе данных ангиографии коронарных артерий
используется подход машинного обучения, который хорошо зарекомендовал себя в сфере компьютерного зрения и обработки изображений. Для решения данной задачи предварительно решили использовать модель Faster-RCNN ResNet-50 из python-фреймворка pytorch:
https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

## Используемый датасет

В процессе подготовки был найден датасет ангиографии коронарных артерий, который представляет собой набор изображений в оттенках серого (1 канал) c разрешением от 512 × 512 до 1000 × 1000 пикселей. Суммарно выборка составила 8325 изображений. Этот датасет мы используем для машинного обучения:
https://data.mendeley.com/datasets/ydrm75xywg/1
