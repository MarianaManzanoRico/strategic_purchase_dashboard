import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def limpiar_datos(Ventas, Clientes, Ramo, Productos):
    Ventas.columns = Ventas.columns.str.strip()
    Ventas = Ventas.rename(columns={
        'Subtotal 1': 'SubTotal1',
        'Importe del impuesto': 'IVA',
        'Base de descuento': 'SubTotal2',
        'An.': 'An',
        'Clase de factura': 'TipoFactura',
        'Fecha factura': 'Fecha',
        'Solic.': 'Solicitante',
        'Material': 'SKU',
        'Número de material': 'Producto',
        'Ctd.facturada': 'CtdFacturada'
    })

    Productos = Productos.rename(columns={'Clave SAP': 'SKU'})

    Clientes = Clientes.rename(columns={
        'Cliente': 'Solicitante',
        'Código de ramo 1': 'Ramo1',
        'Código de ramo 2': 'Ramo2',
        'Código de ramo 3': 'Ramo3',
        'Código de ramo 4': 'Ramo4',
        'Código de ramo 5': 'Ramo5',
        'Clave de ramo industrial': 'Ramo6'
    })

    Ventas['UM'] = Ventas['UM'].replace('CA', 'CJ')
    Ventas = Ventas.drop_duplicates()
    Ventas['UM'] = Ventas['UM'].astype('category')

    LUBCEN = Ventas[
        (Ventas['ClFac'] != 'S2') & (Ventas['ClFac'] != 'ZDP') &
        (~Ventas['UM'].isin(['CUN', 'BOT', 'PZA', 'GA'])) & (Ventas['An'] != 'X')
    ][['Fecha', 'Factura', 'SubTotal1', 'OrgVt', 'Solicitante', 'SKU', 'Producto', 'CtdFacturada', 'UM', 'Volumen']]

    Clientes = Clientes.drop_duplicates()
    LUBCEN['Solicitante'] = LUBCEN['Solicitante'].astype(str)
    Clientes['Solicitante'] = Clientes['Solicitante'].astype(str)

    for col in ['Ramo1', 'Ramo2', 'Ramo3', 'Ramo4', 'Ramo5', 'Ramo6']:
        if col not in Clientes.columns:
            Clientes[col] = "Otro"

    LUBCEN = LUBCEN.merge(Clientes[['Solicitante', 'Ramo1', 'Ramo2', 'Ramo3', 'Ramo4', 'Ramo5', 'Ramo6']], on='Solicitante', how='left')

    Productos['SKU'] = Productos['SKU'].astype(str)
    LUBCEN['SKU'] = LUBCEN['SKU'].astype(str)
    LUBCEN = LUBCEN.merge(Productos[['LOB','SKU']], on='SKU', how='left')

    LUBCEN = LUBCEN.drop_duplicates()

    for col in ['LOB', 'Ramo1', 'Ramo2', 'Ramo3', 'Ramo4', 'Ramo5', 'Ramo6']:
        LUBCEN[col] = LUBCEN[col].fillna("Otro").astype('category')

    LUBCEN['OrgVt'] = LUBCEN['OrgVt'].astype('category')

    LUBCEN['Volumen_total'] = LUBCEN.groupby('SKU')['Volumen'].transform('sum')
    LUBCEN['CtdFact_total'] = LUBCEN.groupby('SKU')['CtdFacturada'].transform('sum')
    LUBCEN['Presentacion'] = LUBCEN['Volumen_total'] / LUBCEN['CtdFact_total']
    LUBCEN = LUBCEN.drop(columns=['Volumen_total', 'CtdFact_total'])

    return LUBCEN

def clasificar_modelos(LUBCEN):
    LUBCEN['Fecha'] = pd.to_datetime(LUBCEN['Fecha'])
    LUBCEN['AñoMes'] = LUBCEN['Fecha'].dt.to_period('M').dt.to_timestamp()
    ultimo_mes = LUBCEN['AñoMes'].max()

    inicio_6m = ultimo_mes - pd.DateOffset(months=5)
    df_6m = LUBCEN[LUBCEN['AñoMes'].between(inicio_6m, ultimo_mes)]

    pedido = df_6m.groupby(['SKU', 'OrgVt', 'Producto', 'UM', 'Presentacion'], observed=True)['Volumen'].sum().reset_index()
    pedido = pedido.sort_values(['OrgVt', 'Volumen'], ascending=[True, False])
    pedido['Pct_Volumen'] = pedido.groupby('OrgVt', observed=True)['Volumen'].transform(lambda x: x / x.sum())
    pedido['Pct_Acum'] = pedido.groupby('OrgVt', observed=True)['Pct_Volumen'].cumsum()
    pedido['Clase_ABC'] = pedido['Pct_Acum'].apply(lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C'))

    inicio_21 = ultimo_mes - pd.DateOffset(months = 23)
    df_21m = LUBCEN[LUBCEN['AñoMes'].between(inicio_21, ultimo_mes)]
    meses_consec = df_21m.groupby(['SKU', 'OrgVt'])['AñoMes'].nunique().reset_index()
    meses_consec['Tipo_pronostico'] = meses_consec['AñoMes'].apply(lambda x: 'ST' if x == 20 else 'PM')

    resultado = pedido.merge(meses_consec[['SKU', 'OrgVt', 'Tipo_pronostico']], on=['SKU', 'OrgVt'], how='left')
    return LUBCEN, resultado


def calcular_inventario_optimo(LUBCEN, resultado):
    LUBCEN['Fecha'] = pd.to_datetime(LUBCEN['Fecha'])
    ventas = LUBCEN.groupby(['SKU', 'OrgVt', 'Fecha'], as_index=False)['Volumen'].sum()
    # fechas_completas = pd.DataFrame({'Fecha': pd.date_range(start='2023-01-01', end='2024-08-31')})
    list_rop = []

    for _, row in ventas[['SKU', 'OrgVt']].drop_duplicates().iterrows():
        df = ventas[(ventas['SKU'] == row['SKU']) & (ventas['OrgVt'] == row['OrgVt'])]
        # df = fechas_completas.merge(df, on='Fecha', how='left')
        df['Volumen'].fillna(0, inplace=True)
        prom = df['Volumen'].mean()
        Lt = prom * 10
        std = df['Volumen'].std(ddof=1)
        rop = prom * 10 + 1.96 * std * np.sqrt(0)
        list_rop.append({
            'SKU': row['SKU'], 'OrgVt': row['OrgVt'],
            'Punto_Reorden': rop,
            'Demanda_LeadTime': Lt
        })

    df_rop = pd.DataFrame(list_rop)
    resultado = resultado.merge(df_rop, on=['SKU', 'OrgVt'], how='left')
    return resultado


def generar_pronosticos(LUBCEN, resultado):
    import math
    df_pm = resultado[resultado['Tipo_pronostico'] == 'PM'].copy()
    ventas_mensuales = LUBCEN.groupby(['SKU', 'OrgVt', pd.Grouper(key='Fecha', freq='M')])['Volumen'].sum().reset_index()
    ventas_pm = ventas_mensuales.merge(df_pm[['SKU', 'OrgVt']], on=['SKU', 'OrgVt'], how='inner')
    ventas_pm = ventas_pm.sort_values(['SKU', 'OrgVt', 'Fecha'])

    def calcular_promedio_movil(grupo):
        grupo['Pronostico_PM'] = grupo['Volumen'].rolling(window=3, min_periods=1).mean()
        return pd.Series({
            'Pronostico_Final': grupo['Pronostico_PM'].iloc[-1],
            'Ultima_Fecha_Historia': grupo['Fecha'].iloc[-1]
        })

    pm = ventas_pm.groupby(['SKU', 'OrgVt']).apply(calcular_promedio_movil).reset_index()
    resultado = resultado.merge(pm, on=['SKU', 'OrgVt'], how='left')

    df_st = resultado[resultado['Tipo_pronostico'] == 'ST']
    LUBCEN['AñoMes'] = LUBCEN['Fecha'].dt.to_period('M').dt.to_timestamp()
    LUB_pronostico = LUBCEN[LUBCEN[['SKU', 'OrgVt']].apply(tuple, axis=1).isin(df_st[['SKU', 'OrgVt']].apply(tuple, axis=1))].copy()
    LUB_pronostico = LUB_pronostico.groupby(['SKU', 'OrgVt', 'AñoMes'])['Volumen'].sum().reset_index()
    LUB_pronostico['AñoMes'] = pd.to_datetime(LUB_pronostico['AñoMes'])

    pronosticos = []
    for _, row in tqdm(LUB_pronostico[['SKU', 'OrgVt']].drop_duplicates().iterrows(), total=len(LUB_pronostico[['SKU', 'OrgVt']].drop_duplicates())):
        df = LUB_pronostico[(LUB_pronostico['SKU'] == row['SKU']) & (LUB_pronostico['OrgVt'] == row['OrgVt'])]
        serie = df.set_index('AñoMes')['Volumen'].astype(float)
        try:
            modelo = auto_arima(serie, seasonal=True, m=12, trace=False, stepwise=True)
            sarima = SARIMAX(serie, order=modelo.order, seasonal_order=modelo.seasonal_order)
            fitted = sarima.fit(disp=False)
            forecast = fitted.get_forecast(steps=2).predicted_mean
            pronosticos.append({
                'SKU': row['SKU'],
                'OrgVt': row['OrgVt'],
                'Volumen_pronosticado': math.ceil(forecast.iloc[1]) 
            })
        except:
            continue

    df_st_pred = pd.DataFrame(pronosticos)
    resultado = resultado.merge(df_st_pred, on=['SKU', 'OrgVt'], how='left')
    resultado['pronostico'] = resultado['Pronostico_Final'].fillna(0) + resultado['Volumen_pronosticado'].fillna(0)
    resultado['pronostico'] = resultado['pronostico'].replace(0, np.nan)

    return resultado.drop(columns=['Pronostico_Final', 'Volumen_pronosticado', 'Ultima_Fecha_Historia'], errors='ignore')