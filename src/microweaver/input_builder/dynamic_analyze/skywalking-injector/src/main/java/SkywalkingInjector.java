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
    
    // TypeSolver reference for determining type source
    private static ReflectionTypeSolver reflectionTypeSolver = null;
    private static List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: java -jar SkywalkingInjector.jar <project-root-path>");
            System.err.println("Example: java -jar SkywalkingInjector.jar D:/Projects/MicroService/traveldog/traveldog");
            System.exit(1);
        }

        String targetProjectRoot = args[0];
        System.out.println("Target project root: " + targetProjectRoot);
        
        // First process all pom.xml files, add SkyWalking dependency
        File projectRootDir = new File(targetProjectRoot);
        System.out.println("Processing pom.xml files...");
        parsePomDirsRecursively(projectRootDir);
        System.out.println("pom.xml files processing completed");
        
        // Recursively find all src/main/java directories (supports multi-module projects)
        List<File> sourceDirs = findAllSourceDirs(projectRootDir);

        // Create CombinedTypeSolver
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        
        // Save ReflectionTypeSolver reference
        reflectionTypeSolver = new ReflectionTypeSolver();
        combinedTypeSolver.add(reflectionTypeSolver);
        
        // Save JavaParserTypeSolver reference
        JavaParserTypeSolver rootJavaParserTypeSolver = new JavaParserTypeSolver(projectRootDir);
        javaParserTypeSolvers.add(rootJavaParserTypeSolver);
        combinedTypeSolver.add(rootJavaParserTypeSolver);
        
        // Add JavaParserTypeSolver for each source directory
        for (File sourceDir : sourceDirs) {
            JavaParserTypeSolver sourceTypeSolver = new JavaParserTypeSolver(sourceDir);
            javaParserTypeSolvers.add(sourceTypeSolver);
            combinedTypeSolver.add(sourceTypeSolver);
        }

        // Collect Java files from all source directories
        List<File> javaFiles = new ArrayList<>();
        for (File sourceDir : sourceDirs) {
            javaFiles.addAll(listJavaFilesRecursively(sourceDir));
        }
        System.out.println("Found " + javaFiles.size() + " Java files");

        // Get available processor count for parallel processing
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Using " + numThreads + " threads for parallel processing");
        
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger processedCount = new AtomicInteger(0);
        
        // Extract file count as final variable for use in lambda
        final int totalFiles = javaFiles.size();
        
        // Submit tasks for each file
        for (File javaFile : javaFiles) {
            executor.submit(() -> {
                try {
                    // Create independent JavaParser and JavaParserFacade instances for each thread
                    // Because JavaParser may not be thread-safe
                    JavaParser parser = new JavaParser();
                    parser.getParserConfiguration().setSymbolResolver(new JavaSymbolSolver(combinedTypeSolver));
                    
                    int current = processedCount.incrementAndGet();
                    if (current % 100 == 0) {
                        System.out.println("Processed " + current + "/" + totalFiles + " files");
                    }
                    
                    addTraceAnnotation(parser, javaFile);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
        
        // Shutdown thread pool and wait for all tasks to complete
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                executor.shutdownNow();
                System.err.println("Warning: processing timeout, force shutting down thread pool");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        System.out.println("All files processing completed");
    }

    /**
     * Recursively find all src/main/java directories under the project root (supports multi-module projects)
     */
    private static List<File> findAllSourceDirs(File rootDir) {
        List<File> sourceDirs = new ArrayList<>();
        findSourceDirsRecursively(rootDir, sourceDirs);
        return sourceDirs;
    }
    
    /**
     * Helper method for recursively finding src/main/java directories
     */
    private static void findSourceDirsRecursively(File dir, List<File> sourceDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory is src/main/java
        if (dir.getName().equals("java") && 
            dir.getParentFile() != null && dir.getParentFile().getName().equals("main") &&
            dir.getParentFile().getParentFile() != null && dir.getParentFile().getParentFile().getName().equals("src")) {
            sourceDirs.add(dir);
            return; // Found, do not go deeper to avoid duplicates
        }
        
        // Recursively search subdirectories
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
     * Recursively list all Java files
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
     * Helper method for recursively finding directories containing pom.xml
     * @param dir Current directory
     * @param pomDirs Result list
     */
    private static void parsePomDirsRecursively(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory contains pom.xml
        File pomFile = new File(dir, "pom.xml");
        if (pomFile.exists() && pomFile.isFile()) {
            addSkyWalkingDependency(pomFile.getAbsolutePath());
        }
        
        // Recursively search subdirectories
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    // Skip common build and dependency directories
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
            // Read pom.xml file
            MavenXpp3Reader reader = new MavenXpp3Reader();
            Model model = reader.read(new FileReader(pomFile));

            // Check if dependency already exists
            boolean dependencyExists = model.getDependencies().stream()
                    .anyMatch(d -> "org.apache.skywalking".equals(d.getGroupId()) &&
                            "apm-toolkit-trace".equals(d.getArtifactId()));

            if (!dependencyExists) {
                // Add new dependency
                Dependency dependency = new Dependency();
                dependency.setGroupId("org.apache.skywalking");
                dependency.setArtifactId("apm-toolkit-trace");
                dependency.setVersion("9.5.0");
                model.addDependency(dependency);

                // Step 3: Write to StringWriter
                StringWriter stringWriter = new StringWriter();
                MavenXpp3Writer xpp3Writer = new MavenXpp3Writer();
                xpp3Writer.write(stringWriter, model);
                String rawXml = stringWriter.toString();

                // Step 4: Parse into DOM
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
//                factory.setIgnoringElementContentWhitespace(true); // Critical!
                Document document = factory.newDocumentBuilder()
                        .parse(new ByteArrayInputStream(rawXml.getBytes("UTF-8")));

                // Step 5: Use Transformer to control indentation output
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
    
    // More precise implementation (process by class)
    private static void addTraceAnnotation(JavaParser parser, File javaFile) {
        try (FileInputStream in = new FileInputStream(javaFile)) {
            CompilationUnit cu = parser.parse(in).getResult().orElseThrow();

            // Check each type declaration, only add annotations to non-JPA entity classes
            for (TypeDeclaration<?> typeDeclaration : cu.findAll(TypeDeclaration.class)) {
                boolean isJPAEntity = typeDeclaration.getAnnotations().stream().anyMatch(annotation -> 
                    "Entity".equals(annotation.getNameAsString()) ||
                    "Data".equals(annotation.getNameAsString()) ||
                    "MappedSuperclass".equals(annotation.getNameAsString()) ||
                    "Embeddable".equals(annotation.getNameAsString())
                );

                // If it is a JPA entity class, skip all methods of this class
                if (isJPAEntity) {
                    continue;
                }

                // Add @Trace annotation to methods of non-JPA entity classes
                for (MethodDeclaration method : typeDeclaration.getMethods()) {
                    if (!method.getAnnotations().stream().anyMatch(a -> a.getNameAsString().equals("Trace"))) {
                        MarkerAnnotationExpr traceAnnotation = new MarkerAnnotationExpr();
                        traceAnnotation.setName("Trace");
                        method.addAnnotation(traceAnnotation);
                    }
                }
            }

            // Check if any @Trace annotations were added, if so add import
            boolean hasTraceMethods = cu.findAll(MethodDeclaration.class).stream()
                .anyMatch(method -> method.getAnnotations().stream()
                    .anyMatch(a -> a.getNameAsString().equals("Trace")));

            if (hasTraceMethods) {
                if (!cu.getImports().stream().anyMatch(i -> i.getNameAsString().equals("org.apache.skywalking.apm.toolkit.trace.Trace"))) {
                    cu.addImport("org.apache.skywalking.apm.toolkit.trace.Trace");
                }
            }

            // Save modified file
            String modifiedCode = new PrettyPrinter().print(cu);
            try (FileWriter writer = new FileWriter(javaFile)) {
                writer.write(modifiedCode);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    
}