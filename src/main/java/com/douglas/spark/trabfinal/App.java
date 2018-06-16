package com.douglas.spark.trabfinal;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.douglas.spark.trabfinal.funcoes.FunParaTuple2;
import com.douglas.spark.trabfinal.leitura.Documento;
import com.douglas.spark.trabfinal.leitura.Documento2;
import com.douglas.spark.trabfinal.leitura.Ideia;
import com.douglas.spark.trabfinal.leitura.Leitor;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoAtribuirPeso;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoCodificacaoASCIIDesnec;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoGenerica;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoMinuscula;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoRemocaoStopWords;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoRemoverTags;
import com.douglas.spark.trabfinal.transformacoes.TransformacaoStemmer;
import com.douglas.spark.trabfinal.utils.DatasetUtils;

import scala.Tuple2;

public class App {
	
	/**
	 * Converte um Dataset<Row> em um Dataset<Tuple2<Object, Object>>.
	 * 
	 * @param dataset
	 *            dataset a ser convertido
	 * @param primeiroElemento
	 *            nome da coluna que será o primeiro elemento da tupla
	 * @param segundoElemento
	 *            nome da coluna que será o segundo elemento da tupla
	 * @return dataset convertido para Tuple2<Object, Object>
	 */
	private static Dataset<Tuple2<Object, Object>> paraTuple2(Dataset<Row> dataset, String primeiroElemento,
			String segundoElemento) {
		@SuppressWarnings("unchecked")
		Class<Tuple2<Object, Object>> tuple2ObjectClasse = (Class<
				Tuple2<Object, Object>>) new Tuple2<Object, Object>(primeiroElemento, segundoElemento).getClass();
		return dataset.map(new FunParaTuple2(DatasetUtils.getIndiceColuna(dataset, primeiroElemento),
				DatasetUtils.getIndiceColuna(dataset, segundoElemento)), Encoders.bean(tuple2ObjectClasse));
	}
	
	public static StringIndexer label() {
		StringIndexer categoryIndexer = new StringIndexer()
			      .setInputCol("classe")
			      .setOutputCol("label");
		return categoryIndexer;
	}
	
	public static Tokenizer tokenizer() {
		Tokenizer tokenizer = new Tokenizer()
				  .setInputCol("coluna_stemmer")
				  .setOutputCol("words");
		return tokenizer;
	}
	
	public static HashingTF hashingTF() {
		HashingTF hashingTF = new HashingTF()
				  .setInputCol("words")
				  .setOutputCol("features");
		return hashingTF;
	}
	
	@SuppressWarnings("rawtypes")
	public static Dataset<Row> transformationTrain(Dataset<Documento2> dataInput) {
		Dataset<Row> dataOutput = dataInput.toDF();
		TransformacaoGenerica teste = null;
	
		try {
						
			teste = TransformacaoMinuscula.class.newInstance().criarTransformacao()
					.setColunaEntrada("revisao")
					.setColunaSaida("coluna_min")
					.setEsquemaEntrada(dataOutput.schema());		
				
			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
			teste = TransformacaoRemoverTags.class.newInstance().criarTransformacao()
					.setColunaEntrada(teste.getColunaSaida())
					.setColunaSaida("coluna_sem_tags")
					.setEsquemaEntrada(dataOutput.schema());
				
			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
			teste = TransformacaoCodificacaoASCIIDesnec.class.newInstance().criarTransformacao()
					.setColunaEntrada(teste.getColunaSaida())
					.setColunaSaida("coluna_sem_cod_asc_desnec")
					.setEsquemaEntrada(dataOutput.schema());
				
			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
			
		
			/*teste = TransformacaoRemocaoStopWords.class.newInstance().criarTransformacao()
					.setColunaEntrada(teste.getColunaSaida())
					.setColunaSaida("coluna_sem_stop")
					.setEsquemaEntrada(dataOutput.schema());
			
//			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);*/

		
			teste = TransformacaoStemmer.class.newInstance().criarTransformacao()
					.setColunaEntrada(teste.getColunaSaida())
					.setColunaSaida("coluna_stemmer")
					.setEsquemaEntrada(dataOutput.schema());
				
			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
			
			teste = TransformacaoAtribuirPeso.class.newInstance().criarTransformacao()
					.setColunaEntrada("classe")
					.setColunaSaida("wght")
					.setEsquemaEntrada(dataOutput.schema());
			
			dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
			
			dataOutput = dataOutput.withColumn("wght", dataOutput.col("wght").cast("double"));
						
		} catch (InstantiationException | IllegalAccessException e) {
			e.printStackTrace();
		}
		return dataOutput;
	}
	
	public static void main(String[] args)  {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		Logger.getLogger("").setLevel(Level.ERROR);
		
	    System.setProperty("spark.executor.memory", "5G");
		
		SparkSession spark = new SparkSession
				.Builder()
				.appName("TrabalhoFinal")
				.master("local[*]")
				.getOrCreate();
		
		// Teste para leitura do dataset
		Dataset<Documento2> dataSet = Leitor.lerDeDoc("dataset1.csv");
		Dataset<Row> dataRow = App.transformationTrain(dataSet);
		Dataset<Row>[] splits = dataRow.randomSplit(new double[]{0.8, 0.2}, 1234L);
		Dataset<Row> training = splits[0].toDF();
		Dataset<Row> test = splits[1].toDF();
				
		/* 
		 * LogisticRegression lr = new LogisticRegression()
				.setWeightCol("wght")
				.setLabelCol("label");*/
		
		LinearSVC ls = new LinearSVC()
				.setLabelCol("label");
		
		StringIndexer categoryIndexer = new StringIndexer()
			      .setInputCol("classe")
			      .setOutputCol("label");
		
		Tokenizer tokenizer = new Tokenizer()
			  .setInputCol("coluna_stemmer")
			  .setOutputCol("words");
		
		NGram ngram = new NGram()
				.setN(1)
				.setInputCol("words")
				.setOutputCol("ngrams");
		
		HashingTF hashingTF = new HashingTF()
			  .setInputCol("ngrams")			  
			  .setOutputCol("col_htf");
		
		/*Word2Vec w2V = new Word2Vec()
				  .setInputCol("ngrams")
				  .setOutputCol("features")
				  .setNumPartitions(1)
				  .setSeed(0)
				  .setMinCount(5);*/
				
		IDF idf = new IDF()
				.setInputCol(hashingTF.getOutputCol())
				.setOutputCol("features")
				.setMinDocFreq(0);
		
		/*ChiSqSelector sel = new ChiSqSelector()
				  .setNumTopFeatures(6850)
				  .setFeaturesCol("col_htf")
				  .setLabelCol("label")
				  .setOutputCol("features");*/
				
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {categoryIndexer, tokenizer, ngram, hashingTF, idf, ls});
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction");
		
		// Parâmetros do svm
		ParamMap[] paramGrid = new ParamGridBuilder()
				 .addGrid(hashingTF.numFeatures(), new int[] {16384})
				 .addGrid(ls.regParam(), new double[] {1.0})
				 .addGrid(ls.aggregationDepth(),new int[] {2})
				 .addGrid(ls.maxIter(),new int[] {500})
				 .addGrid(ls.threshold(),new double[] {0.0})
				 .addGrid(ls.tol(), new double[] {1E-6})
				 .build();
		
		/*
		 // Parâmetros do Logistic Regression
		 * ParamMap[] paramGrid = new ParamGridBuilder()
				 .addGrid(hashingTF.numFeatures(), new int[] {14700})
				 .addGrid(lr.regParam(), new double[] {0.0001})
				 .addGrid(lr.elasticNetParam(), new double[] {0.1})
				 .addGrid(lr.aggregationDepth(),new int[] {2})
				 .addGrid(lr.maxIter(),new int[] {60})
				 .addGrid(lr.standardization())
				 .addGrid(lr.fitIntercept())
				 .addGrid(lr.threshold(),new double[] {0.5})
				 .addGrid(lr.tol(), new double[] {1E-6})
				 .build();*/
						
		String nomeDoModelo = "modeloCV_00";
		
		CrossValidatorModel cvModel = null;
				
		String predicoesNome = "predicoes_" + nomeDoModelo;
		File diretorioModelo = new File(nomeDoModelo);
		File diretorioPredicoes = new File(predicoesNome);
		if (diretorioPredicoes.exists() && diretorioModelo.exists()) {
			cvModel = CrossValidatorModel.read().load(nomeDoModelo);
		} else {
			if (diretorioModelo.exists()) {
				diretorioModelo.delete();
			}
			if (diretorioPredicoes.exists()) {
				diretorioPredicoes.delete();
			}
			
			try {
				CrossValidator cv = new CrossValidator().setEstimator(pipeline)
						.setEstimatorParamMaps(paramGrid)
						.setEvaluator(evaluator)
						.setNumFolds(5);
			
				cvModel = cv.fit(training);
				//cvModel.transform(training).show(false);
							
				cvModel.write().save(nomeDoModelo);		
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
				
		Dataset<Row> predicoes;

		if (new File(predicoesNome).exists()) {
			predicoes = spark.read().load(predicoesNome);
		} else {
			predicoes = cvModel.bestModel().transform(test);
			predicoes.write().save(predicoesNome);
		}
				
		predicoes.show();
		
		Dataset<Tuple2<Object, Object>> predicoesTuple2 = paraTuple2(predicoes, "prediction", "label");
		
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predicoesTuple2.rdd());
		
		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
				
		System.out.println(cvModel.getEstimatorParamMaps());
		System.out.println("\nConfusion matrix: \n" + confusion);

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
			System.out.format("\n\nClass %f precision = %f\n", metrics.labels()[i],
					metrics.precision(metrics.labels()[i]));
			System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics.labels()[i]));
			System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(metrics.labels()[i]));
		}

		// Weighted stats
		System.out.println("\n\nMétricas calculadas com org.apache.spark.mllib.evaluation.MulticlassMetrics\n");
		System.out.format("Accuracy = %f\n", metrics.accuracy());
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		System.out.format("Weighted true positive rate = %f\n", metrics.weightedTruePositiveRate());
				
		System.out
				.println("\n\nMétricas calculadas org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator\n");
		MulticlassClassificationEvaluator mEvaluator = new MulticlassClassificationEvaluator();
		System.out.println("F1 -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("accuracy");
		System.out.println("Accuracy -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("weightedPrecision");
		System.out.println("weightedPrecision -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("weightedRecall");
		System.out.println("weightedRecall -> " + mEvaluator.evaluate(predicoes));

		System.out.println(
				"\nmedia das métricas F1 para cross validator em treino -> " + Arrays.toString(cvModel.avgMetrics()));	

	}
}
