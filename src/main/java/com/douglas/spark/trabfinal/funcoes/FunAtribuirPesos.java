package com.douglas.spark.trabfinal.funcoes;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class FunAtribuirPesos implements MapFunction<Row, Row> {

	private static final long serialVersionUID = 1L;

	/**
	 * Coluna em que deve aplicar o pré-processamento.
	 */
	private int colunaEntrada;
	/**
	 * Coluna em que deve salvar string após o pré-processamento.
	 */
	private int colunaSaida;
	
	public FunAtribuirPesos(int colunaEntrada, int colunaSaida) {
		this.colunaEntrada = colunaEntrada;
		this.colunaSaida = colunaSaida;		
	}
	
	public String calculateWeights(double label, String classe) {
		
		if(classe.equals("1.0")) {
			return Double.toString(1 * label);
		}else {
			return Double.toString(1 * (1.0 - label));
		}
	}
	
	public double balanceDataset() {
		double numPositivo = 149.00;// 149.00
		double datasetSize = 1499.00;// 1499.00
		double balancingRatio = (datasetSize - numPositivo) / datasetSize;
		return balancingRatio;
	}

	/**
	 * Aplica a conversão da string em letras minúsculas salva na coluna de saída e
	 * retorna nova linha.
	 */
	@Override
	public Row call(Row row) throws Exception {
		int n = row.length();
		if (n == colunaSaida) {
			n++;
		}
		Object[] campos = new Object[n];
		for (int i = 0; i < row.length(); i++) {
			campos[i] = row.get(i);
		}
				
		campos[colunaSaida] = calculateWeights(balanceDataset(),row.getString(colunaEntrada));
		return RowFactory.create(campos);
	}
}