package com.douglas.spark.trabfinal.leitura;

public class Documento {
	private String id;
	private String rating;
	private String revisao;
	private String classe; 
	
	public Documento(String id, String classEstrela, String revisao, String classe) {
		this.id = id;
		this.rating = classEstrela;
		this.revisao = revisao;
		this.classe = classe;
	}
	
	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getClassRating() {
		return rating;
	}

	public void setRating(String rating) {
		this.rating = rating;
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
		return id + "\t" + rating + "\t" + classe + "\t" + revisao;
	}
	
	
}
