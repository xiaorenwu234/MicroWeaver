import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.expr.MarkerAnnotationExpr;
import com.github.javaparser.printer.PrettyPrinter;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;

import org.apache.maven.model.Dependency;
import org.apache.maven.model.Model;
import org.apache.maven.model.io.xpp3.MavenXpp3Reader;
import org.apache.maven.model.io.xpp3.MavenXpp3Writer;
import org.w3c.dom.Document;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

public class SkywalkingInjector {
    
    // TypeSolver 引用，用于判断类型来源
    private static ReflectionTypeSolver reflectionTypeSolver = null;
    private static List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("用法: java -jar SkywalkingInjector.jar <项目根目录路径>");
            System.err.println("示例: java -jar SkywalkingInjector.jar D:/Projects/MicroService/traveldog/traveldog");
            System.exit(1);
        }

        String targetProjectRoot = args[0];
        System.out.println("目标项目根目录: " + targetProjectRoot);
        
        // 首先处理所有 pom.xml 文件，添加 SkyWalking 依赖
        File projectRootDir = new File(targetProjectRoot);
        System.out.println("开始处理 pom.xml 文件...");
        parsePomDirsRecursively(projectRootDir);
        System.out.println("pom.xml 文件处理完成");
        
        // 递归查找所有 src/main/java 目录（支持多模块项目）
        List<File> sourceDirs = findAllSourceDirs(projectRootDir);

        // 创建 CombinedTypeSolver
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        
        // 保存 ReflectionTypeSolver 引用
        reflectionTypeSolver = new ReflectionTypeSolver();
        combinedTypeSolver.add(reflectionTypeSolver);
        
        // 保存 JavaParserTypeSolver 引用
        JavaParserTypeSolver rootJavaParserTypeSolver = new JavaParserTypeSolver(projectRootDir);
        javaParserTypeSolvers.add(rootJavaParserTypeSolver);
        combinedTypeSolver.add(rootJavaParserTypeSolver);
        
        // 为每个源代码目录添加 JavaParserTypeSolver
        for (File sourceDir : sourceDirs) {
            JavaParserTypeSolver sourceTypeSolver = new JavaParserTypeSolver(sourceDir);
            javaParserTypeSolvers.add(sourceTypeSolver);
            combinedTypeSolver.add(sourceTypeSolver);
        }

        // 从所有源代码目录中收集 Java 文件
        List<File> javaFiles = new ArrayList<>();
        for (File sourceDir : sourceDirs) {
            javaFiles.addAll(listJavaFilesRecursively(sourceDir));
        }
        System.out.println("找到 " + javaFiles.size() + " 个 Java 文件");

        // 获取可用处理器数量，用于并行处理
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("使用 " + numThreads + " 个线程进行并行处理");
        
        // 创建线程池
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger processedCount = new AtomicInteger(0);
        
        // 将文件总数提取为final变量，以便在lambda中使用
        final int totalFiles = javaFiles.size();
        
        // 为每个文件提交任务
        for (File javaFile : javaFiles) {
            executor.submit(() -> {
                try {
                    // 为每个线程创建独立的 JavaParser 和 JavaParserFacade 实例
                    // 因为 JavaParser 可能不是线程安全的
                    JavaParser parser = new JavaParser();
                    parser.getParserConfiguration().setSymbolResolver(new JavaSymbolSolver(combinedTypeSolver));
                    
                    int current = processedCount.incrementAndGet();
                    if (current % 100 == 0) {
                        System.out.println("已处理 " + current + "/" + totalFiles + " 个文件");
                    }
                    
                    addTraceAnnotation(parser, javaFile);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
        
        // 关闭线程池并等待所有任务完成
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                executor.shutdownNow();
                System.err.println("警告: 处理超时，强制关闭线程池");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        System.out.println("所有文件处理完成");
    }

    /**
     * 递归查找项目根目录下所有的 src/main/java 目录（支持多模块项目）
     */
    private static List<File> findAllSourceDirs(File rootDir) {
        List<File> sourceDirs = new ArrayList<>();
        findSourceDirsRecursively(rootDir, sourceDirs);
        return sourceDirs;
    }
    
    /**
     * 递归查找 src/main/java 目录的辅助方法
     */
    private static void findSourceDirsRecursively(File dir, List<File> sourceDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // 检查当前目录是否是 src/main/java
        if (dir.getName().equals("java") && 
            dir.getParentFile() != null && dir.getParentFile().getName().equals("main") &&
            dir.getParentFile().getParentFile() != null && dir.getParentFile().getParentFile().getName().equals("src")) {
            sourceDirs.add(dir);
            return; // 找到后不再深入，避免重复
        }
        
        // 递归查找子目录
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    findSourceDirsRecursively(file, sourceDirs);
                }
            }
        }
    }

    /**
     * 递归列出所有 Java 文件
     */
    private static List<File> listJavaFilesRecursively(File dir) {
        List<File> javaFiles = new ArrayList<>();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    javaFiles.addAll(listJavaFilesRecursively(file));
                } else if (file.getName().endsWith(".java")) {
                    javaFiles.add(file);
                }
            }
        }
        return javaFiles;
    }

    /**
     * 递归查找包含 pom.xml 的目录的辅助方法
     * @param dir 当前目录
     * @param pomDirs 结果列表
     */
    private static void parsePomDirsRecursively(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // 检查当前目录是否包含 pom.xml
        File pomFile = new File(dir, "pom.xml");
        if (pomFile.exists() && pomFile.isFile()) {
            addSkyWalkingDependency(pomFile.getAbsolutePath());
        }
        
        // 递归查找子目录
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    // 跳过常见的构建和依赖目录
                    String dirName = file.getName();
                    if (!dirName.equals("target") && 
                        !dirName.equals(".git") && 
                        !dirName.equals(".svn") &&
                        !dirName.equals("node_modules") &&
                        !dirName.equals(".idea") &&
                        !dirName.equals(".vscode") &&
                        !dirName.equals("microweaver-static-dependency")) {
                        parsePomDirsRecursively(file);
                    }
                }
            }
        }
    }
    
    public static void addSkyWalkingDependency(String pomPath) {
        File pomFile = new File(pomPath);
        try {
            // 读取 pom.xml 文件
            MavenXpp3Reader reader = new MavenXpp3Reader();
            Model model = reader.read(new FileReader(pomFile));

            // 检查是否已经有相同的依赖
            boolean dependencyExists = model.getDependencies().stream()
                    .anyMatch(d -> "org.apache.skywalking".equals(d.getGroupId()) &&
                            "apm-toolkit-trace".equals(d.getArtifactId()));

            if (!dependencyExists) {
                // 添加新的依赖
                Dependency dependency = new Dependency();
                dependency.setGroupId("org.apache.skywalking");
                dependency.setArtifactId("apm-toolkit-trace");
                dependency.setVersion("9.5.0");
                model.addDependency(dependency);

                // Step 3: 写入 StringWriter
                StringWriter stringWriter = new StringWriter();
                MavenXpp3Writer xpp3Writer = new MavenXpp3Writer();
                xpp3Writer.write(stringWriter, model);
                String rawXml = stringWriter.toString();

                // Step 4: 解析成 DOM
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
//                factory.setIgnoringElementContentWhitespace(true); // 关键！
                Document document = factory.newDocumentBuilder()
                        .parse(new ByteArrayInputStream(rawXml.getBytes("UTF-8")));

                // Step 5: 使用 Transformer 控制缩进写出
                Transformer transformer = TransformerFactory.newInstance().newTransformer();
                transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8");

                DOMSource source = new DOMSource(document);
                StreamResult result = new StreamResult(new FileWriter(pomFile));
                transformer.transform(source, result);

                System.out.println("Dependency added successfully.");
            } else {
                System.out.println("Dependency already exists.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // 更精确的实现（按类处理）
    private static void addTraceAnnotation(JavaParser parser, File javaFile) {
        try (FileInputStream in = new FileInputStream(javaFile)) {
            CompilationUnit cu = parser.parse(in).getResult().orElseThrow();

            // 检查每个类型声明，只对非 JPA 实体类添加注解
            for (TypeDeclaration<?> typeDeclaration : cu.findAll(TypeDeclaration.class)) {
                boolean isJPAEntity = typeDeclaration.getAnnotations().stream().anyMatch(annotation -> 
                    "Entity".equals(annotation.getNameAsString()) ||
                    "Data".equals(annotation.getNameAsString()) ||
                    "MappedSuperclass".equals(annotation.getNameAsString()) ||
                    "Embeddable".equals(annotation.getNameAsString())
                );

                // 如果是 JPA 实体类，跳过这个类的所有方法
                if (isJPAEntity) {
                    continue;
                }

                // 对非 JPA 实体类的方法添加 @Trace 注解
                for (MethodDeclaration method : typeDeclaration.getMethods()) {
                    if (!method.getAnnotations().stream().anyMatch(a -> a.getNameAsString().equals("Trace"))) {
                        MarkerAnnotationExpr traceAnnotation = new MarkerAnnotationExpr();
                        traceAnnotation.setName("Trace");
                        method.addAnnotation(traceAnnotation);
                    }
                }
            }

            // 检查是否添加了任何 @Trace 注解，如果是则添加 import
            boolean hasTraceMethods = cu.findAll(MethodDeclaration.class).stream()
                .anyMatch(method -> method.getAnnotations().stream()
                    .anyMatch(a -> a.getNameAsString().equals("Trace")));

            if (hasTraceMethods) {
                if (!cu.getImports().stream().anyMatch(i -> i.getNameAsString().equals("org.apache.skywalking.apm.toolkit.trace.Trace"))) {
                    cu.addImport("org.apache.skywalking.apm.toolkit.trace.Trace");
                }
            }

            // 保存修改后的文件
            String modifiedCode = new PrettyPrinter().print(cu);
            try (FileWriter writer = new FileWriter(javaFile)) {
                writer.write(modifiedCode);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    
}