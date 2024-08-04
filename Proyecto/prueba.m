% % %Red neuronal
% % 
% % %Limpiar la pantalla
%  clear, clc, close all
% % 
% % % Ruta del archivo CSV
% % % Aquí se debe de poner el archivo que estamos evaluando
% % %Dataset 1
% % archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Chile.csv"; 
% % %Dataset 2
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\data.csv";
% % %Dataset 3
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\database.csv";
% % %Dataset 4
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\earthquake.csv";
% % %Dataset 5
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Earthquakes_v2.csv";
% % %Dataset 6
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Mexico.csv";
% % 
% % % 
% % % %archivoCSV = 'C:\Users\danie\Downloads\breast-cancer.csv';
% % % % Leer el archivo CSV utilizando readtable
% % datos = readtable(archivoCSV);
% % % 
% % % datos
% % % 
% % % % Obtener los datos de la tabla 'datos'
% % % matrizDatos = table2array(datos);
% % % 
% % % 
% % % 
% % % % Calcular las medias y desviaciones estándar de cada columna
% % % medias = mean(matrizDatos);
% % % desviaciones = std(matrizDatos);
% % % 
% % % % Normalizar los datos utilizando la normalización Z-score
% % % datosNormalizados = (matrizDatos - medias) ./ desviaciones;
% % % 
% % % % Crear una nueva tabla con los datos normalizados
% % % datosNormalizadosTabla = array2table(datosNormalizados, 'VariableNames', datos.Properties.VariableNames);
% % % 
% % % % Mostrar los datos normalizados
% % % disp(datosNormalizadosTabla);
% % % 
% % % 
% % % 
% % % % % Obtener las columnas de la tabla 'datos'
% % % % columnas = datos.Properties.VariableNames;
% % % % 
% % % % % Discretizar cada columna en dos categorías y asignar valores binarios (0 y 1)
% % % % for i = 1:numel(columnas)
% % % %     columna = datos.(columnas{i});
% % % %     categorias = discretize(columna, 2, 'categorical');
% % % %     categorias = double(categorias) - 1; % Convertir las categorías en valores numéricos (0 y 1)
% % % %     datos.(columnas{i}) = categorias; % Reemplazar la columna original con las categorías discretizadas
% % % % end
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % % Convertir las variables numéricas a categóricas
% % % %datos = convertvars(datos, {'GF','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','MaxPd','MaxV2d','MaxV3d','MaxM2d','MaxM3d','MaxTd','MaxPc','MaxV2c','MaxV3c','MaxM2c','MaxM3c','MaxTc','U1','U2','U3','R1','R2','R3','W'}, 'categorical');
% % % 
% % % % Aplicar rmoutliers a la tabla data
% % % %dataSinOutliers = datosNormalizadosTabla;
% % % dataSinOutliers = rmoutliers(datos);
% % % 
% % % % Mostrar la tabla resultante sin outliers
% % % disp(dataSinOutliers);
% % % 
% % % 
% % % 
% % % 
% % % % Obtener los datos de la tabla 'Datos'
% % % %datos = app.Datos; % Suponiendo que tus datos se almacenan en la variable 'Datos' de tu App Designer
% % % 
% % % % Convertir la tabla en una matriz
% % % matrizDatos = table2array(dataSinOutliers);
% % % 
% % % % Separar los datos en características (X) y etiquetas (Y)
% % % X = matrizDatos(:, 1:end-1); % Características (todas las columnas excepto la última)
% % % Y = matrizDatos(:, end); % Etiquetas (última columna)
% % % 
% % % 
% % % 
% % % 
% % % 
% % % % Dividir los datos en conjuntos de entrenamiento y prueba
% % % porcentajeEntrenamiento = 0.9; % Porcentaje de datos para entrenamiento (80%)
% % % indicesEntrenamiento = randperm(size(X, 1), round(porcentajeEntrenamiento * size(X, 1)));
% % % indicesPrueba = setdiff(1:size(X, 1), indicesEntrenamiento);
% % % 
% % % X_train = X(indicesEntrenamiento, :); % Conjunto de entrenamiento (características)
% % % Y_train = Y(indicesEntrenamiento, :); % Conjunto de entrenamiento (etiquetas)
% % % 
% % % X_test = X(indicesPrueba, :); % Conjunto de prueba (características)
% % % Y_test = Y(indicesPrueba, :); % Conjunto de prueba (etiquetas)
% % % 
% % % % Crear y entrenar la red neuronal
% % % hiddenLayerSize = 200; % Número de neuronas en la capa oculta
% % % net = fitnet(hiddenLayerSize); % Crear la red neuronal
% % % net = train(net, X_train', Y_train'); % Entrenar la red neuronal
% % % 
% % % % Evaluar la red neuronal en los datos de prueba
% % % Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba
% % % 
% % % % Calcular la precisión de la red neuronal
% % % accuracy = sum(Y_pred' == Y_test) / length(Y_test);
% % % 
% % % % Mostrar la precisión en la pantalla
% % % disp(['Precisión de la red neuronal: ' num2str(accuracy)]);
% % % 
% % % 
% % % 
% % % 
% % % 
% % % % Dividir los datos en características (X) y etiquetas (Y)
% % % X = matrizDatos(:, 1:end-1); % Características (todas las columnas excepto la última)
% % % Y = matrizDatos(:, end); % Etiquetas (última columna)
% % % 
% % % % Definir los parámetros de la validación cruzada
% % % numFolds = 10; % Número de particiones para la validación cruzada
% % % 
% % % % Inicializar el vector para almacenar las precisiones de cada partición
% % % accuracy = zeros(numFolds, 1);
% % % 
% % % % Realizar la validación cruzada
% % % cv = cvpartition(size(X, 1), 'KFold', numFolds);
% % % 
% % % for i = 1:numFolds
% % %     % Obtener los conjuntos de entrenamiento y prueba para esta partición
% % %     trainIdx = training(cv, i);
% % %     testIdx = test(cv, i);
% % %     X_train = X(trainIdx, :);
% % %     Y_train = Y(trainIdx, :);
% % %     X_test = X(testIdx, :);
% % %     Y_test = Y(testIdx, :);
% % % 
% % %     % Crear y entrenar la red neuronal
% % %     hiddenLayerSize = 200; % Número de neuronas en la capa oculta
% % %     net = fitnet(hiddenLayerSize); % Crear la red neuronal
% % %     net = train(net, X_train', Y_train'); % Entrenar la red neuronal
% % %     
% % %     % Evaluar la red neuronal en los datos de prueba
% % %     Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba
% % %     
% % %     % Calcular la precisión de la red neuronal
% % %     accuracy(i) = sum(Y_pred' == Y_test) / length(Y_test);
% % % end
% % % 
% % % % Calcular la precisión promedio
% % % averageAccuracy = mean(accuracy);
% % % 
% % % % Mostrar la precisión promedio
% % % disp(['Precisión promedio de la red neuronal con validación cruzada: ' num2str(averageAccuracy)]);
% % % 
% % % 
% % 
% % 
% % 
% % 
% % 
% % % Leer los datos desde un archivo o tenerlos previamente cargados en una variable llamada 'datos'
% % %datos = readtable("D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Chile.csv");
% % 
% % % Obtener los datos de interés para el preprocesamiento y la red neuronal (características y etiquetas)
% % X = datos(:, 1:end-1); % Características (todas las columnas excepto la última)
% % Y = datos(:, end); % Etiquetas (última columna)
% % 
% % % Convertir las características y etiquetas a formato numérico
% % X = table2array(X);
% % Y = table2array(Y);
% % 
% % % Verificar y eliminar valores faltantes (NaN) en las características
% % missingData = any(isnan(X), 2);
% % X(missingData, :) = [];
% % Y(missingData, :) = [];
% % 
% % 
% % % Calcular el límite de los outliers usando el rango intercuartil (IQR)
% % Q1 = prctile(X, 25);
% % Q3 = prctile(X, 75);
% % IQR = Q3 - Q1;
% % limiteInferior = Q1 - 1.5 * IQR;
% % limiteSuperior = Q3 + 1.5 * IQR;
% % 
% % % Eliminar los outliers de cada columna de características
% % for i = 1:size(X, 2)
% %     columna = X{:, i};
% %     columnaSinOutliers = columna(columna >= limiteInferior(i) & columna <= limiteSuperior(i));
% %     X{:, i} = columnaSinOutliers;
% % end
% % 
% % % Eliminar las filas correspondientes en las etiquetas
% % Y(X{:, 1} < limiteInferior(1) | X{:, 1} > limiteSuperior(1), :) = [];
% % 
% % % Convertir las características y etiquetas a formato numérico
% % X = table2array(X);
% % Y = table2array(Y);
% % 
% % % Dividir los datos en conjuntos de entrenamiento y prueba
% % porcentajeEntrenamiento = 0.8; % Porcentaje de datos para entrenamiento (80%)
% % indicesEntrenamiento = randperm(size(X, 1), round(porcentajeEntrenamiento * size(X, 1)));
% % indicesPrueba = setdiff(1:size(X, 1), indicesEntrenamiento);
% % 
% % X_train = X(indicesEntrenamiento, :); % Conjunto de entrenamiento (características)
% % Y_train = Y(indicesEntrenamiento, :); % Conjunto de entrenamiento (etiquetas)
% % 
% % X_test = X(indicesPrueba, :); % Conjunto de prueba (características)
% % Y_test = Y(indicesPrueba, :); % Conjunto de prueba (etiquetas)
% % 
% % % Crear y entrenar la red neuronal
% % hiddenLayerSize = 100; % Número de neuronas en la capa oculta
% % net = fitnet(hiddenLayerSize); % Crear la red neuronal
% % net = train(net, X_train', Y_train'); % Entrenar la red neuronal
% % 
% % % Evaluar la red neuronal en los datos de prueba
% % Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba
% % 
% % % Calcular la precisión de la red neuronal
% % accuracy = sum(Y_pred' == Y_test) / length(Y_test);
% % 
% % % Mostrar la precisión en la pantalla
% % disp(['Precisión de la red neuronal: ' num2str(accuracy)]);
% % 
% % % Clasificar nuevos datos utilizando la red neuronal
% % nuevosDatos = readtable('nuevos_datos.csv'); % Leer los nuevos datos desde un archivo
% % nuevasCaracteristicas = nuevosDatos(:, 1:end-1); % Obtener las características de los nuevos datos
% % nuevasCaracteristicas = table2array(nuevasCaracteristicas); % Convertir las características a formato numérico
% % 
% % % Predecir las etiquetas de los nuevos datos utilizando la red neuronal
% % nuevasEtiquetas = net(nuevasCaracteristicas'); % Predecir las etiquetas de los nuevos datos
% % 
% % % Mostrar las etiquetas predichas en la pantalla
% % disp('Etiquetas predichas para los nuevos datos:');
% % disp(nuevasEtiquetas);
% 
% 
% 
% 
% % % Aquí se debe de poner el archivo que estamos evaluando
% % %Dataset 1
% % archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Chile.csv"; 
% % %Dataset 2
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\data.csv";
% % %Dataset 3
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\database.csv";
% % %Dataset 4
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\earthquake.csv";
% % %Dataset 5
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Earthquakes_v2.csv";
% % %Dataset 6
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Mexico.csv";
% 
% 
% clear, clc, close all
% 
% % Ruta del archivo CSV
% archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\data.csv"; % Reemplaza 'ruta/al/archivo.csv' con la ruta de tu archivo CSV
% %archivoCSV = 'C:\Users\danie\Downloads\breast-cancer.csv';
% % Leer el archivo CSV utilizando readtable
% datos = readtable(archivoCSV);
% 
% datos
% 
% % Aplicar rmoutliers a la tabla data
% dataSinOutliers = rmoutliers(datos);
% 
% % Mostrar la tabla resultante sin outliers
% disp(dataSinOutliers);
% 
% 
% 
% 
% % Obtener los datos de la tabla 'Datos'
% %datos = app.Datos; % Suponiendo que tus datos se almacenan en la variable 'Datos' de tu App Designer
% 
% % Convertir la tabla en una matriz
% matrizDatos = table2array(dataSinOutliers);
% 
% % Separar los datos en características (X) y etiquetas (Y)
% X = matrizDatos(:, 1:end-1); % Características (todas las columnas excepto la última)
% Y = matrizDatos(:, end); % Etiquetas (última columna)
% 
% 
% 
% 
% 
% % Dividir los datos en conjuntos de entrenamiento y prueba
% porcentajeEntrenamiento = 0.9; % Porcentaje de datos para entrenamiento (80%)
% indicesEntrenamiento = randperm(size(X, 1), round(porcentajeEntrenamiento * size(X, 1)));
% indicesPrueba = setdiff(1:size(X, 1), indicesEntrenamiento);
% 
% X_train = X(indicesEntrenamiento, :); % Conjunto de entrenamiento (características)
% Y_train = Y(indicesEntrenamiento, :); % Conjunto de entrenamiento (etiquetas)
% 
% X_test = X(indicesPrueba, :); % Conjunto de prueba (características)
% Y_test = Y(indicesPrueba, :); % Conjunto de prueba (etiquetas)
% 
% % Crear y entrenar la red neuronal
% hiddenLayerSize = 100; % Número de neuronas en la capa oculta
% net = fitnet(hiddenLayerSize); % Crear la red neuronal
% net = train(net, X_train', Y_train'); % Entrenar la red neuronal
% 
% % Evaluar la red neuronal en los datos de prueba
% Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba
% 
% % Calcular la precisión de la red neuronal
% accuracy = sum(Y_pred' == Y_test) / length(Y_test);
% 
% % Mostrar la precisión en la pantalla
% disp(['Precisión de la red neuronal: ' num2str(accuracy * 100)]);





% % %Dataset 1
% % archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Chile.csv"; 
% % %Dataset 2
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\data.csv";
% % %Dataset 3
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\database.csv";
% % %Dataset 4
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\earthquake.csv";
% % %Dataset 5
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Earthquakes_v2.csv";
% % %Dataset 6
% % %archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Mexico.csv";

clear, clc, close all

% Ruta del archivo CSV
archivoCSV = "D:\dmr_escuela\dmr_23a\dmr_23a_cdld\datasets_proyecto\Chile.csv"; % Reemplaza 'ruta/al/archivo.csv' con la ruta de tu archivo CSV
%archivoCSV = 'C:\Users\danie\Downloads\breast-cancer.csv';
% Leer el archivo CSV utilizando readtable
datos = readtable(archivoCSV);

datos

% Obtener los datos de la tabla 'datos'
matrizDatos = table2array(datos);

% Calcular las medias y desviaciones estándar de cada columna
medias = mean(matrizDatos);
desviaciones = std(matrizDatos);

% Normalizar los datos utilizando la normalización Z-score
datosNormalizados = (matrizDatos - medias) ./ desviaciones;

% Crear una nueva tabla con los datos normalizados
datosNormalizadosTabla = array2table(datosNormalizados, 'VariableNames', datos.Properties.VariableNames);

% Mostrar los datos normalizados
disp(datosNormalizadosTabla);



% % Obtener las columnas de la tabla 'datos'
% columnas = datos.Properties.VariableNames;
% 
% % Discretizar cada columna en dos categorías y asignar valores binarios (0 y 1)
% for i = 1:numel(columnas)
%     columna = datos.(columnas{i});
%     categorias = discretize(columna, 2, 'categorical');
%     categorias = double(categorias) - 1; % Convertir las categorías en valores numéricos (0 y 1)
%     datos.(columnas{i}) = categorias; % Reemplazar la columna original con las categorías discretizadas
% end








% Convertir las variables numéricas a categóricas
%datos = convertvars(datos, {'GF','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','MaxPd','MaxV2d','MaxV3d','MaxM2d','MaxM3d','MaxTd','MaxPc','MaxV2c','MaxV3c','MaxM2c','MaxM3c','MaxTc','U1','U2','U3','R1','R2','R3','W'}, 'categorical');

% Aplicar rmoutliers a la tabla data
dataSinOutliers = datosNormalizadosTabla;

% Mostrar la tabla resultante sin outliers
disp(dataSinOutliers);




% Obtener los datos de la tabla 'Datos'
%datos = app.Datos; % Suponiendo que tus datos se almacenan en la variable 'Datos' de tu App Designer

% Convertir la tabla en una matriz
matrizDatos = table2array(dataSinOutliers);

% Separar los datos en características (X) y etiquetas (Y)
X = matrizDatos(:, 1:end-1); % Características (todas las columnas excepto la última)
Y = matrizDatos(:, end); % Etiquetas (última columna)





% Dividir los datos en conjuntos de entrenamiento y prueba
porcentajeEntrenamiento = 0.8; % Porcentaje de datos para entrenamiento (80%)
indicesEntrenamiento = randperm(size(X, 1), round(porcentajeEntrenamiento * size(X, 1)));
indicesPrueba = setdiff(1:size(X, 1), indicesEntrenamiento);

X_train = X(indicesEntrenamiento, :); % Conjunto de entrenamiento (características)
Y_train = Y(indicesEntrenamiento, :); % Conjunto de entrenamiento (etiquetas)

X_test = X(indicesPrueba, :); % Conjunto de prueba (características)
Y_test = Y(indicesPrueba, :); % Conjunto de prueba (etiquetas)

% Crear y entrenar la red neuronal
hiddenLayerSize = 100; % Número de neuronas en la capa oculta
net = fitnet(hiddenLayerSize); % Crear la red neuronal
net = train(net, X_train', Y_train'); % Entrenar la red neuronal

% Evaluar la red neuronal en los datos de prueba
Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba

% Calcular la precisión de la red neuronal
accuracy = sum(Y_pred' == Y_test) / length(Y_test);

% Mostrar la precisión en la pantalla
disp(['Precisión de la red neuronal: ' num2str(accuracy)]);





% Dividir los datos en características (X) y etiquetas (Y)
X = matrizDatos(:, 1:end-1); % Características (todas las columnas excepto la última)
Y = matrizDatos(:, end); % Etiquetas (última columna)

% Definir los parámetros de la validación cruzada
numFolds = 5; % Número de particiones para la validación cruzada

% Inicializar el vector para almacenar las precisiones de cada partición
accuracy = zeros(numFolds, 1);

% Realizar la validación cruzada
cv = cvpartition(size(X, 1), 'KFold', numFolds);

for i = 1:numFolds
    % Obtener los conjuntos de entrenamiento y prueba para esta partición
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx, :);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx, :);

    % Crear y entrenar la red neuronal
    hiddenLayerSize = 100; % Número de neuronas en la capa oculta
    net = fitnet(hiddenLayerSize); % Crear la red neuronal
    net = train(net, X_train', Y_train'); % Entrenar la red neuronal
    
    % Evaluar la red neuronal en los datos de prueba
    Y_pred = net(X_test'); % Predecir las etiquetas en el conjunto de prueba
    
    % Calcular la precisión de la red neuronal
    accuracy(i) = sum(Y_pred' == Y_test) / length(Y_test);
end

% Calcular la precisión promedio
averageAccuracy = mean(accuracy);

% Mostrar la precisión promedio
disp(['Precisión promedio de la red neuronal con validación cruzada: ' num2str(averageAccuracy)]);


% 
% % Obtener las etiquetas predichas por la red neuronal
% Y_pred = net(X_test');
% 
% % Convertir las etiquetas a valores numéricos (0 y 1)
% Y_pred_numeric = round(Y_pred);
% Y_test_numeric = round(Y_test);
% 
% % Crear la matriz de confusión
% C = confusionmat(Y_test_numeric, Y_pred_numeric);
% 
% % Visualizar la matriz de confusión
% figure;
% heatmap(C, 'Colormap', parula(2), 'ColorbarVisible', 'off', 'XLabel', 'Predicción', 'YLabel', 'Etiqueta Verdadera');
% title('Matriz de Confusión');



% Obtener las etiquetas predichas por la red neuronal
Y_pred = net(X_test');

% Convertir las etiquetas a valores numéricos (0 y 1)
Y_pred_numeric = round(Y_pred);
Y_test_numeric = round(Y_test);

% Crear la matriz de confusión
C = confusionmat(Y_test_numeric, Y_pred_numeric);

% Mostrar la matriz de confusión como números
disp('Matriz de Confusión:');
disp(C);


% Obtener las etiquetas predichas por la red neuronal
Y_pred = net(X_test');

% Convertir las etiquetas a valores numéricos (0 y 1)
Y_pred_numeric = round(Y_pred);
Y_test_numeric = round(Y_test);

% Crear la matriz de confusión
C = confusionmat(Y_test_numeric, Y_pred_numeric);

% Extraer los valores de TP, FP, TN y FN
TP = C(2, 2);
FP = C(1, 2);
TN = C(1, 1);
FN = C(2, 1);

% Mostrar los valores de TP, FP, TN y FN
disp('Matriz de Confusión:');
disp(['Verdaderos Positivos (TP): ' num2str(TP)]);
disp(['Falsos Positivos (FP): ' num2str(FP)]);
disp(['Verdaderos Negativos (TN): ' num2str(TN)]);
disp(['Falsos Negativos (FN): ' num2str(FN)]);


% alpha = 0.05; % Nivel de significancia
% n = length(Y_test); % Tamaño de la muestra
% accuracy_model1 = accuracy;
% accuracy_model2 = 0.90;
% diff_accuracy = accuracy_model1 - accuracy_model2;
% std_error_diff = sqrt((accuracy_model1 * (1 - accuracy_model1))/n + (accuracy_model2 * (1 - accuracy_model2))/n);
% t_critical = tinv(1 - alpha/2, n-1);
% lower_bound = diff_accuracy - t_critical * std_error_diff;
% upper_bound = diff_accuracy + t_critical * std_error_diff;
% p_value = 2 * (1 - tcdf(abs(diff_accuracy/std_error_diff), n-1));
% 
% disp(['Precisión del modelo 1: ' num2str(accuracy_model1)]);
% disp(['Precisión del modelo 2: ' num2str(accuracy_model2)]);
% disp(['Diferencia de precisión: ' num2str(diff_accuracy)]);
% disp(['Error estándar de la diferencia: ' num2str(std_error_diff)]);
% disp(['Valor crítico de la prueba t: ' num2str(t_critical)]);
% disp(['Intervalo de confianza para la diferencia de precisión: [' num2str(lower_bound) ', ' num2str(upper_bound) ']']);
% disp(['Valor p de la prueba t: ' num2str(p_value)]);
