package com.douglas.spark.trabfinal.leitura;

public class Documento2 {
	private String id;
	private String revisao;
	private String classe; 
	
	public Documento2(String id, String classe,String revisao) {
		this.id = id;
		this.classe = classe;
		this.revisao = revisao;
	}
	
	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getRevisao() {
		return revisao;
	}

	public void setRevisao(String revisao) {
		this.revisao = revisao;
	}
	
	public String getClasse() {
		return classe;
	}

	public void setClasse(String classe) {
		this.classe = classe;
	}

	@Override
	public String toString() {
		return id + "\t" + classe + "\t" + revisao;
	}
	
	
}
