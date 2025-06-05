
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from lubcen_modular import limpiar_datos, clasificar_modelos, calcular_inventario_optimo, generar_pronosticos
import plotly.graph_objects as go
import time
import psutil

@st.cache_data(show_spinner=False)
def load_all_data(ventas_file, productos_file):
    xls = pd.ExcelFile(ventas_file)
    Ventas = pd.read_excel(xls, sheet_name="Zfact")
    Clientes = pd.read_excel(xls, sheet_name="Zclientes")
    Ramo = pd.read_excel(xls, sheet_name="MCSI")
    Productos = pd.read_excel(productos_file)
    return limpiar_datos(Ventas, Clientes, Ramo, Productos)

st.set_page_config(page_title="Dashboard de Compras Estratégicas Lubcen", layout="wide")
t0 = time.perf_counter()

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 6])
with col_logo:
    logo = Image.open("lubcen_logo.png")
    st.image(logo, width=90)
with col_title:
    st.title("Dashboard de Compras Estratégicas – Lubcen / Mobil")

st.sidebar.header("Sube tus archivos")
ventas_file       = st.sidebar.file_uploader("Excel Ventas (Zfact, Zclientes, MCSI)", type=['xlsx'])
productos_file    = st.sidebar.file_uploader("Excel Productos", type=['xlsx'])
inventarios_file  = st.sidebar.file_uploader("Excel Inventario Actual", type=['xlsx'])

if ventas_file and productos_file:
    LUBCEN = load_all_data(ventas_file, productos_file)

    # Filtros
    with st.sidebar:
        st.header("Filtros")

        if 'Ramo6' in LUBCEN.columns and pd.api.types.is_categorical_dtype(LUBCEN['Ramo6']):
            division = st.multiselect("División de consumidor", LUBCEN['Ramo6'].cat.categories.to_list())
        else:
            division = []

        if 'LOB' in LUBCEN.columns and pd.api.types.is_categorical_dtype(LUBCEN['LOB']):
            linea = st.multiselect("Línea de negocio (LOB)", LUBCEN['LOB'].cat.categories.to_list())
        else:
            linea = []

        if 'OrgVt' in LUBCEN.columns and pd.api.types.is_categorical_dtype(LUBCEN['OrgVt']):
            region = st.multiselect("Región", LUBCEN['OrgVt'].cat.categories.to_list())
        else:
            region = []

        if st.button("Resetear filtros"):
            division = linea = region = []


        condiciones = []
        if 'Ramo6' in LUBCEN.columns and division:
            condiciones.append(LUBCEN['Ramo6'].isin(division))
        if 'LOB' in LUBCEN.columns and linea:
            condiciones.append(LUBCEN['LOB'].isin(linea))
        if 'OrgVt' in LUBCEN.columns and region:
            condiciones.append(LUBCEN['OrgVt'].isin(region))

        # Combinar condiciones solo si hay al menos una
        if condiciones:
            from functools import reduce
            import operator
            filtro_combinado = reduce(operator.and_, condiciones)
            df_filtered = LUBCEN[filtro_combinado]
        else:
            df_filtered = LUBCEN.copy()
        st.write(f"Número de registros después de aplicar filtros: {len(df_filtered)}")

    st.divider()
    st.subheader("Indicadores generales")

    df_f2 = df_filtered.copy()
    if 'Fecha' in df_f2.columns:
        df_f2['Fecha'] = pd.to_datetime(df_f2['Fecha'])
    else:
        st.error("La columna 'Fecha' no está disponible en los datos. Revisa el archivo de ventas.")
        st.stop()
    df_f2['Mes'] = df_f2['Fecha'].dt.to_period('M')
    ultimo_anio = df_f2['Fecha'].dt.year.max()
    df_ultimo_anio = df_f2[df_f2['Fecha'].dt.year == ultimo_anio]

    volumen_total = df_ultimo_anio['Volumen'].sum()
    volumen_mensual = df_f2.groupby('Mes')['Volumen'].sum()
    variacion_mensual = ((volumen_mensual.iloc[-1] - volumen_mensual.iloc[-2]) / volumen_mensual.iloc[-2]) * 100 if len(volumen_mensual) >= 2 else 0

    k1, k2 = st.columns(2)
    k1.metric("Volumen total vendido (L)", f"{volumen_total:,.0f}")
    k2.metric("Variación mensual", f"{variacion_mensual:.1f}%")

    st.divider()
    st.subheader("Top productos y líneas")

    top_productos = df_ultimo_anio.groupby('Producto')['Volumen'].sum().nlargest(10).reset_index()
    top_lineas = df_ultimo_anio.groupby('LOB')['Volumen'].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(top_productos, x='Volumen', y='Producto', orientation='h', title="Top 10 productos por volumen", color_discrete_sequence=["#0066B3"], category_orders={'Producto': top_productos['Producto'].tolist()})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(top_lineas, names='LOB', values='Volumen', title="Distribución por línea de negocio", hole=0.4, color_discrete_sequence=["#0063af", "#e40428", "#404759", "#eceff1"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    df_time = df_filtered.copy()
    df_time['Fecha'] = pd.to_datetime(df_time['Fecha'])
    df_time.set_index('Fecha', inplace=True)
    vol_resampled = df_time['Volumen'].resample('M').mean()
    with col3:
        fig3 = go.Figure(go.Scatter(x=vol_resampled.index, y=vol_resampled.values, mode='lines'))
        fig3.update_layout(title="Volumen Vendido Promediado por Mes", xaxis_title='Fecha', yaxis_title='Volumen Vendido')
        st.plotly_chart(fig3, use_container_width=True)

    vol_por_um = df_ultimo_anio.groupby('UM')['Volumen'].sum().reset_index().rename(columns={'Volumen': 'Volumen total'})
    with col4:
        fig4 = px.bar(vol_por_um, x='UM', y='Volumen total', title="Volumen vendido por UM", color_discrete_sequence=["#0066B3"])
        st.plotly_chart(fig4, use_container_width=True)

    # Recomendaciones
    st.divider()
    st.subheader("Recomendaciones de compra")
    if inventarios_file:
        resultado = clasificar_modelos(LUBCEN)[1]
        resultado = calcular_inventario_optimo(LUBCEN, resultado)
        resultado = generar_pronosticos(LUBCEN, resultado)

        inventario_df = pd.read_excel(inventarios_file)
        inventario_df['Material'] = inventario_df['Material'].astype(str)
        resultado['SKU'] = resultado['SKU'].astype(str)
        inventario_df = inventario_df.rename(columns={
            'Material': 'SKU',
            'Org Vtas': 'OrgVt'
        })

        merged = resultado.merge(
            inventario_df[['Material', 'Org Vtas', 'Inventario Lts.']],
            on=['SKU', 'OrgVt'],
            how='left'
        )
        merged['Inventario Lts.'] = merged['Inventario Lts.'].fillna(0)
        merged['Recomendaciones'] = merged['pronostico'] + merged['Punto_Reorden'] - merged['Inventario Lts.'] - merged['Demanda_LeadTime']

        st.dataframe(merged[['SKU', 'OrgVt', 'Producto', 'pronostico', 'Punto_Reorden', 'Inventario Lts.', 'Recomendaciones']], use_container_width=True)

    else:
        st.warning("Por favor sube el archivo de inventarios.")

else:
    st.warning("Por favor, sube los archivos de ventas y productos.")

# Sidebar performance
t1 = time.perf_counter()
tiempo = t1 - t0
cpu = psutil.cpu_percent(interval=1)
ram = psutil.virtual_memory().percent
st.sidebar.header("📊 Métricas de Rendimiento")
st.sidebar.metric("CPU utilizada", f"{cpu:.1f} %")
st.sidebar.metric("RAM utilizada", f"{ram:.1f} %")
st.sidebar.metric("Tiempo de carga", f"{tiempo:.2f} s")