package com.douglas.spark.trabfinal.transformacoes;

import com.douglas.spark.trabfinal.funcoes.FunAtribuirPesos;

public class TransformacaoAtribuirPeso extends TransformacaoGenerica<FunAtribuirPesos> {

	private static final long serialVersionUID = 1L;
	
	@Override
	protected String getLabeluid() {
		return "TransformacaoAtribuirPeso";
	}

	@Override
	public TransformacaoGenerica<FunAtribuirPesos> criarTransformacao() {
		return new TransformacaoAtribuirPeso();
	}

	@Override
	public FunAtribuirPesos criarFuncao(int indiceColEntrada, int indiceColSaida) {
		return new FunAtribuirPesos(indiceColEntrada, indiceColSaida);
	}
}
