package com.douglas.spark.trabfinal.leitura;

public class Ideia {
	
	private String classe;
	private String sentenca;
	 
	public Ideia(String classe,String sentenca) {
		this.classe = classe;
		this.sentenca = sentenca;
	}


	public String getClasse() {
		return classe;
	}


	public void setClasse(String classe) {
		this.classe = classe;
	}


	public String getSentenca() {
		return sentenca;
	}


	public void setSentenca(String sentenca) {
		this.sentenca = sentenca;
	}
	
	@Override
	public String toString() {
		return classe + "\t" + sentenca;
	}
	
}
