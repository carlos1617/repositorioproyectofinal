from flask import Flask,render_template,request
app = Flask(__name__)
@app.route("/")
def inicio():
    return render_template('inicio.html')
@app.route("/pronosticos")
def pronosticos():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg 
    
    datos = {'fecha':['1/4/2022', '2/4/2022', '3/4/2022', '4/4/2022', 
                  '5/4/2022', '6/4/2022', '7/4/2022', '8/4/2022', '9/4/2022', '10/4/2022'],
                'Ventas_diarias':[100, 120, 180, 90, 135, 210, 153, 120, 140, 170],
                'Gastos_diarios':[40, 45, 100, 50, 60, 72, 60, 83, 34, 62]}
    data = pd.DataFrame(datos).to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    

    datos1 = {'fecha':['1/4/2022', '2/4/2022', '3/4/2022', '4/4/2022', 
                    '5/4/2022', '6/4/2022', '7/4/2022', '8/4/2022', '9/4/2022', '10/4/2022'],
    'Ventas_diarias':[100, 120, 180, 90, 135, 210, 153, 120, 140, 170]}
    #print(datos1)
    movil = pd.DataFrame(datos1)
    moviles = movil.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    # mostramos los 5 primeros registros
    print(moviles)

    # calculamos para la primera media móvil MMO_3
    for i in range(0,movil.shape[0]-2):
        movil.loc[movil.index[i+2],'MMO_3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        
    # calculamos para la segunda media móvil MMO_4
    for i in range(0,movil.shape[0]-3):
        movil.loc[movil.index[i+3],'MMO_4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
    3,1])/4),1)
        
    # calculamos la proyeción final
    proyeccion = movil.iloc[7:,[1,2,3]]
    p1,p2,p3 =proyeccion.mean()

    # incorporamos al DataFrame
    a = movil.append({'fecha':'11/4/2022','Ventas_diarias':p1, 'MMO_3':p2, 'MMO_4':p3},ignore_index=True)
    # mostramos los resultados
    a['e_MM3'] = a['Ventas_diarias']-a['MMO_3']
    a['e_MM4'] = a['Ventas_diarias']-a['MMO_4']
    a
    movillle=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    
    plt.grid(True)
    plt.plot(a['Ventas_diarias'],label='Ventas_diarias',marker='o')
    plt.plot(a['MMO_3'],label='Media Móvil 10/4/2022')
    plt.plot(a['MMO_4'],label='Media Móvil 11/4/2022')
    plt.legend(loc=2)
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')


    movil = pd.DataFrame(datos1)
    
    # mostramos los 5 primeros registros
    
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'fecha':'11/4/2022','Ventas_diarias':p1, 'SN':p2},ignore_index=True)
    print(a)
    tabla2 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    
    a = pd.DataFrame(datos1)
    x = a.index.values
    y= a["Ventas_diarias"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
    p0,p1 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1)


    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    y_ajuste = p[0]*x + p[1]
    print (y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper left")
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
   
    plot_url1 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    p = np.polyfit(x,y,2)
    p1,p2,p3 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
    
    y_ajuste = p[0]*x*x + p[1]*x + p[2]
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
   
    plot_url2 = base64.b64encode(buf.getvalue()).decode('UTF-8')
    
    n=x.size
    x1 = []
    x2 = []
    for i in [12,13]:
        y1_ajuste = p[0]*i*i + p[1]*i + p[2]
        print (f" z = {i} w = {y1_ajuste}")
        x1.append(i)
        x2.append(y1_ajuste)
        
    a["y_ajuste"]=y_ajuste

    dp = pd.DataFrame({'fecha':['11/4/2022','12/4/2022'], 'Ventas_diarias':[0,0],'y_ajuste':x2})
    dp
    a = a.append(dp,ignore_index=True)
    print(a)
    tabla3 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    x = a.index.values
    y_ajuste = a["y_ajuste"]
    y= a["Ventas_diarias"]
    p_datos =plt.plot(y,'b.')
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('tasa Pasiva Referencial')
    plt.axvspan(0,9,alpha=0.3,color='y') # encajonamos los datos iniciales
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
   
    plot_url3 = base64.b64encode(buf.getvalue()).decode('UTF-8')


    datosnew = {'fecha':['1/4/2022', '2/4/2022', '3/4/2022', '4/4/2022', 
                    '5/4/2022', '6/4/2022', '7/4/2022', '8/4/2022', '9/4/2022', '10/4/2022'],
    'Ventas_diarias':[100, 120, 180, 90, 135, 210, 153, 120, 140, 170],
    'Gastos_diarios':[40, 45, 100, 50, 60, 72, 60, 83, 34, 62]}

    datosnew=pd.DataFrame(datosnew).to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    datos3 = {'fecha':['1/4/2022', '2/4/2022', '3/4/2022', '4/4/2022', 
                    '5/4/2022', '6/4/2022', '7/4/2022', '8/4/2022', '9/4/2022', '10/4/2022'],
    'Gastos_diarios':[40, 45, 100, 50, 60, 72, 60, 83, 34, 62]}


    movil = pd.DataFrame(datos3)
    moviless = movil.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    # mostramos los 5 primeros registros
    print(moviless)

    # calculamos para la primera media móvil MMO_3
    for i in range(0,movil.shape[0]-2):
        movil.loc[movil.index[i+2],'MMO_3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        
    # calculamos para la segunda media móvil MMO_4
    for i in range(0,movil.shape[0]-3):
        movil.loc[movil.index[i+3],'MMO_4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
    3,1])/4),1)
        
    # calculamos la proyeción final
    proyeccion = movil.iloc[7:,[1,2,3]]
    p1,p2,p3 =proyeccion.mean()

    # incorporamos al DataFrame
    a = movil.append({'fecha':'11/4/2022','Gastos_diarios':p1, 'MMO_3':p2, 'MMO_4':p3},ignore_index=True)
    # mostramos los resultados
    a['e_MM3'] = a['Gastos_diarios']-a['MMO_3']
    a['e_MM4'] = a['Gastos_diarios']-a['MMO_4']
    a
    movilll=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    plt.grid(True)
    plt.plot(a['Gastos_diarios'],label='Gastos_diarios',marker='o')
    plt.plot(a['MMO_3'],label='Media Móvil de 10/4/2022')
    plt.plot(a['MMO_4'],label='Media Móvil de 11/4/2022')
    plt.legend(loc=2)

    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url14 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    movil = pd.DataFrame(datos3)
    # mostramos los 5 primeros registros
    movil.head()
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'fecha':'11/4/2022','Gastos_diarios':p1, 'SN':p2},ignore_index=True)
    novil=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    a = pd.DataFrame(datos3)
    x = a.index.values
    y= a["Gastos_diarios"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
    p0,p1 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    y_ajuste = p[0]*x + p[1]
    print (y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper left")
   

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    
    plot_url6 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    p = np.polyfit(x,y,2)
    p1,p2,p3 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
    
    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    y_ajuste = p[0]*x*x + p[1]*x + p[2]
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    
    plot_url7 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    n=x.size
    x1 = []
    x2 = []
    for i in [12,13]:
        y1_ajuste = p[0]*i*i + p[1]*i + p[2]
        print (f" z = {i} w = {y1_ajuste}")
        x1.append(i)
        x2.append(y1_ajuste)
        
    a["y_ajuste"]=y_ajuste

    dp = pd.DataFrame({'fecha':['11/4/2022','12/4/2022'], 'Gastos_diarios':[0,0],'y_ajuste':x2})
    dp
    a = a.append(dp,ignore_index=True)
    novile=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    x = a.index.values
    y_ajuste = a["y_ajuste"]
    y= a["Gastos_diarios"]
    p_datos =plt.plot(y,'b.')
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('tasa Pasiva Referencial')
    plt.axvspan(0,9,alpha=0.3,color='y') # encajonamos los datos iniciales
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    
    plot_url8 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    return render_template('pronostico.html', data=data, movillle=movillle, movil=moviles, image=plot_url, tabla2=tabla2, image2=plot_url1, image3=plot_url2, tabla3=tabla3,
    image4=plot_url3, datosnew=datosnew, movill=movilll, image5=plot_url14, novil=novil, image6=plot_url6, image7=plot_url7, novile=novile, image8=plot_url8 ) 
@app.route("/montecarlo")
def montecarlo():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg 

    datos = {
        'id':['1', '2', '3', '4', '5', '1', '2', '3', '4','5'],
        'dia':['lunes', 'martes', 'miercole', 'jueves', 'viernes', 'lunes', 'martes', 'miercoles', 'jueves','viernes'],
        'fecha':['1/4/2022', '2/4/2022', '3/4/2022', '4/4/2022', 
                    '5/4/2022', '6/4/2022', '7/4/2022', '8/4/2022', '9/4/2022', '10/4/2022'],
    'Ventas_diarias':[100, 120, 180, 90, 135, 210, 153, 120, 140, 170],
    'Gastos_diarios':[40, 45, 100, 50, 60, 72, 60, 83, 34, 62]}
    dato = pd.DataFrame(datos)
    data2=dato.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    demanda = dato.filter(items=["id","dia", "fecha", "Ventas_diarias"])
    deman=demanda.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    demanda = demanda.groupby("dia")
    demanda.sum()
    mean = demanda.mean()
    demada=pd.DataFrame(mean)
    deman1=demada.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    suma = mean['Ventas_diarias'].sum()
    n=len(mean)
    suma
    x1 = mean.assign(Probabilidad=lambda x: x['Ventas_diarias'] / suma)
    x2 = x1.sort_values('dia')
    a=x2['Probabilidad']
    a = pd.DataFrame(a)
    deman2=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    a1= np.cumsum(a) #Cálculo la suma acumulativa de las probabilidades
    x2['FPA'] =a1
    x2 = pd.DataFrame(x2)
    deman3=x2.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    x2['Min'] = x2['FPA']
    x2['Max'] = x2['FPA']
    x2
    deman4=x2.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    lis = x2["Min"].values
    lis2 = x2['Max'].values
    lis[0]= 0
    for i in range(1,6):
        lis[i] = lis2[i-1]
        print(i,i-1)
    x2['Min'] = lis
    x2
    # Borland C/C++ xi+1=22695477xi + 1 mod 2^32
    n, m, a, x0, c = 52, 2**32, 22695477, 4, 1
    x = [1] * n
    r = [0.1] * n
    for i in range(0, n):
        x[i] = ((a*x0)+c) % m
        x0 = x[i]
        r[i] = x0 / m
    # llenamos nuestro DataFrame 
    d = {'ri': r }
    dfMCL = pd.DataFrame(data=d)
    dfMCL

    deman5=dfMCL.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    max = x2 ['Max'].values
    min = x2 ['Min'].values
    print(min)
    print(max)
    def busqueda(arrmin, arrmax, valor):
    #print(valor)
        for i in range (len(arrmin)):
        # print(arrmin[i],arrmax[i])
            if valor >= arrmin[i] and valor <= arrmax[i]:
                return i
                print(i)
        return -1
    xpos = dfMCL['ri']
    posi = [0] * n
    print (n)
    for j in range(n):
        val = xpos[j]
        pos = busqueda(min,max,val)
        posi[j] = pos
    x2 = x2.astype({'Ventas_diarias' : int })
    x2
    deman6=x2.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    return render_template('montecarlo.html', data2=data2, deman=deman, deman1=deman1, deman2=deman2, deman3=deman3, deman4=deman4, deman5=deman5, deman6=deman6 ) 
@app.route('/estadistica', methods=['GET', 'POST'])
def mediamm():
    if request.method == 'POST':
        file = request.files['file'].read()
        tipoArch = request.form.get("tipoarchivo")
        columna = request.form.get("nombreColumna")

        # importamos la libreria Pandas, matplotlib y numpy que van a ser de mucha utilidad para poder hacer gráficos
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import io
        from io import BytesIO
        import base64
        from pandas import DataFrame
        # leemos los datos de la tabla del directorio Data de trabajo
        if tipoArch == '1':

            datos = pd.read_excel(file)

        elif tipoArch == '2':
            datos = pd.read_csv(io.StringIO(file.decode('utf-8')))

        elif tipoArch == '3':
            datos = pd.read_json(file)

        elif tipoArch == '4':
            datos = pd.read_html(file)
        elif tipoArch == '5':
            datos = pd.read_clipboard(file)

        elif tipoArch == '6':
            datos = pd.read_feather(file)

        elif tipoArch == '7':
            datos = pd.read_fwf(file)

        elif tipoArch == '8':
            datos = pd.read_gbq(file)

        elif tipoArch == '9':
            datos = pd.read_parquet(file)

        elif tipoArch == '10':
            datos = pd.read_pickle(file)

        elif tipoArch == '11':
            datos = pd.read_msgpack(file)

        elif tipoArch == '12':
            datos = pd.read_sas(file)

        elif tipoArch == '13':
            datos = pd.read_sql(file)

        elif tipoArch == '14':
            datos = pd.read_sql_query(file)
        elif tipoArch == '15':
            datos = pd.read_sql_table(file)

        # Presentamos los datos en un DataFrame de Pandas
        datos

        # Preparando para el grafico para la columna TOTAL PACIENTES
        buf = io.BytesIO()
        x = datos[columna]
        plt.figure(figsize=(10, 5))
        plt.hist(x, bins=8, color='blue')
        plt.axvline(x.mean(), color='red', label='Media')
        plt.axvline(x.median(), color='yellow', label='Mediana')
        plt.axvline(x.mode()[0], color='green', label='Moda')
        plt.xlabel('Total de datos')
        plt.ylabel('Frecuencia')
        plt.legend()

        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

        media = datos[columna].mean()
        moda = datos[columna].mode()
        mediana = datos[columna].median()

        df = pd.DataFrame(columns=('Media', 'Moda', 'Mediana'))
        df.loc[len(df)] = [media, moda, mediana]
        df
        data = df.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

        # Tomamos los datos de las columnas
        df2 = datos[[columna]].describe()
        # describe(), nos presenta directamente la media, desviación standar, el valor mínimo, valor máximo, el 1er cuartil, 2do Cuartil, 3er Cuartil
        data2 = df2.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

        return render_template('mediamm.html', data=data, data2=data2, image=plot_url)
    return render_template('mediamm.html')
@app.route('/CuadradosMedios', methods=['GET', 'POST'])
def CuadradosMedios():
    if request.method == 'POST':
        n = request.form.get('numeroIteraciones', type=int)
        r = request.form.get('semilla', type=int)
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pandas import ExcelWriter
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import io
        from io import BytesIO
        import base64
        l = len(str(r))
        lista = []
        lista2 = []
        i = 1
        while i <= n:
            x = str(r*r)
            if l % 2 == 0:
                x = x.zfill(l*2)
            else:
                x = x.zfill(l)
            y = (len(x)-l)/2
            y = int(y)
            r = int(x[y:y+l])
            lista.append(r)
            lista2.append(x)
            i = i+1
        df = pd.DataFrame({'X2': lista2, 'Xi': lista})
        dfrac = df["Xi"]/10**l
        df['ri'] = dfrac
        buf = io.BytesIO()
        x1 = df['ri']
        plt.plot(x1)
        plt.title('Generador de Números Aleatorios Cuadrados Medios')
        plt.xlabel('Serie')
        plt.ylabel('Aleatorios')
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')
        data = df.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
        return render_template('CuadradosMedios.html', data=data, image=plot_url)
    return render_template('CuadradosMedios.html')
@app.route('/congruencialineal', methods=['GET', 'POST'])
def CongruencialLineal():
    if request.method == 'POST':
        n = request.form.get("numeroIteraciones", type=int)
        x0 = request.form.get("semilla", type=int)
        a = request.form.get("multiplicador", type=int)
        c = request.form.get("incremento", type=int)
        m = request.form.get("modulo", type=int)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from pandas import ExcelWriter
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import io
        from io import BytesIO
        import base64
        #n, m, a, x0, c = 20,1000,101,4,457
        x = [1]*n
        r = [0.1]*n
        for i in range(0, n):
            x[i] = ((a*x0)+c) % m
            x0 = x[i]
            r[i] = x0/m
        df = pd.DataFrame({'Xn': x, 'ri': r})
        # Graficamos los numeros generados
        buf = io.BytesIO()
        plt.plot(r, marker='o')
        plt.title('Generador de Números Aleatorios Congruencial Lineal')
        plt.xlabel('Serie')
        plt.ylabel('Aleatorios')
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')
        data = df.to_html(classes="table table-hover table-striped",
                          justify="justify-all", border=2)
        return render_template('CongruencialLineal.html', data=data, image=plot_url)
    return render_template('CongruencialLineal.html')
@app.route('/CongruencialMultiplicativo', methods=['GET', 'POST'])
def CongruencialMultiplicativo():
    if request.method == 'POST':
        n = request.form.get("numeroIteraciones", type=int)
        x0 = request.form.get("semilla", type=int)
        a = request.form.get("multiplicador", type=int)
        m = request.form.get("modulo", type=int)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from pandas import ExcelWriter
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import io
        from io import BytesIO
        import base64
        x = [1] * n
        r = [0.1] * n
        for i in range(0, n):
            x[i] = (a*x0) % m
            x0 = x[i]
            r[i] = x0 / m
        d = {'Xn': x, 'ri': r}
        df = pd.DataFrame(data=d)
        buf = io.BytesIO()
        plt.plot(r, 'g-', marker='o',)
        plt.title('Generador de Números Aleatorios Congruencial Multiplicativo')
        plt.xlabel('Serie')
        plt.ylabel('Aleatorios')
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')
        data = df.to_html(classes="table table-hover table-striped",
                          justify="justify-all", border=2)
        return render_template('CongruencialMultiplicativo.html', data=data, image=plot_url)
    return render_template('CongruencialMultiplicativo.html')
@app.route("/maual")
def manual():
    return render_template('manual.html')
@app.route("/video")
def video():
    return render_template('video.html')
if __name__ == '__main__':
    app.run(port=5000,debug=True)