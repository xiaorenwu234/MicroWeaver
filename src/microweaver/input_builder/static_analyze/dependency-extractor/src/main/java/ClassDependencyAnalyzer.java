import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.MethodReferenceExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.WildcardType;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.VoidType;
import com.github.javaparser.Position;
import com.github.javaparser.resolution.Context;
import com.github.javaparser.resolution.UnsolvedSymbolException;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedAnnotationDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedFieldDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedReferenceTypeDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedTypeDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedValueDeclaration;
import com.github.javaparser.resolution.types.ResolvedArrayType;
import com.github.javaparser.resolution.types.ResolvedReferenceType;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.resolution.types.ResolvedWildcard;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFactory;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.resolution.model.SymbolReference;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * 类级别依赖分析器
 * 分析类之间的依赖关系，包括：
 * - 字段类型依赖
 * - 方法参数类型依赖
 * - 方法返回类型依赖
 * - 继承关系
 * - 接口实现关系
 * - 方法内部使用的类型
 */
public class ClassDependencyAnalyzer {
    
    /**
     * 类信息数据结构，统一存储类的所有信息
     */
    public static class ClassInfo {
        private String qualifiedName;           // 类的完整限定名
        private String filePath;                // 文件相对路径
        private String typeKind;                // 类型：class、interface、enum、annotation
        private List<String> dependencies;      // 依赖列表
        private List<String> extendsAndImplements; // 继承和实现关系
        private String javaDoc;                 // JavaDoc信息
        private List<String> methods;          // 方法签名列表
        
        public ClassInfo() {
            this.dependencies = new ArrayList<>();
            this.extendsAndImplements = new ArrayList<>();
            this.methods = new ArrayList<>();
            this.javaDoc = "";
        }
        
        // Getters and Setters
        public String getQualifiedName() { return qualifiedName; }
        public void setQualifiedName(String qualifiedName) { this.qualifiedName = qualifiedName; }
        
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        
        public String getTypeKind() { return typeKind; }
        public void setTypeKind(String typeKind) { this.typeKind = typeKind; }
        
        public List<String> getDependencies() { return dependencies; }
        public void setDependencies(List<String> dependencies) { this.dependencies = dependencies; }
        
        public List<String> getExtendsAndImplements() { return extendsAndImplements; }
        public void setExtendsAndImplements(List<String> extendsAndImplements) { this.extendsAndImplements = extendsAndImplements; }
        
        public String getJavaDoc() { return javaDoc; }
        public void setJavaDoc(String javaDoc) { this.javaDoc = javaDoc; }
        
        public List<String> getMethods() { return methods; }
        public void setMethods(List<String> methods) { this.methods = methods; }
    }
    
    // 项目包前缀，用于过滤只保留项目内的类依赖
    private static String projectPackagePrefix = null;

    private static Boolean collectAllDependencies = false;
    
    // 错误信息收集器（线程安全）
    private static final ConcurrentLinkedQueue<String> errorMessages = new ConcurrentLinkedQueue<>();
    
    // TypeSolver 引用，用于判断类型来源
    private static ReflectionTypeSolver reflectionTypeSolver = null;
    private static List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();
    private static List<JarTypeSolver> jarTypeSolvers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        String targetProjectRoot = args[0];
        System.out.println("目标项目根目录: " + targetProjectRoot);

        String classInfoPath = args[1];
        System.out.println("输出路径: " + classInfoPath);

        if (args.length > 2) {
            projectPackagePrefix = args[2];
            System.out.println("项目包前缀: " + projectPackagePrefix);
        }

        // 递归查找所有 src/main/java 目录（支持多模块项目）
        File projectRootDir = new File(targetProjectRoot);
        List<File> sourceDirs = findAllSourceDirs(projectRootDir);
        
        if (sourceDirs.isEmpty()) {
            System.err.println("错误: 在项目根目录中未找到任何 src/main/java 目录");
            System.exit(1);
        }
        
        System.out.println("找到 " + sourceDirs.size() + " 个源代码目录:");
        for (File dir : sourceDirs) {
            System.out.println("  - " + dir.getAbsolutePath());
        }

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

        // deleteDependencyDirsRecursively(projectRootDir);

        loadStaticDependencies(combinedTypeSolver, projectRootDir);
        System.out.println("静态依赖加载完成");

        // 从所有源代码目录中收集 Java 文件
        List<File> javaFiles = new ArrayList<>();
        for (File sourceDir : sourceDirs) {
            javaFiles.addAll(listJavaFilesRecursively(sourceDir));
        }
        System.out.println("找到 " + javaFiles.size() + " 个 Java 文件");

        // if (javaFiles.size() > 200) {
        //     javaFiles = javaFiles.subList(0, 200);
        // }

        // 统一存储类信息: 类名 -> ClassInfo（使用线程安全的集合）
        Map<String, ClassInfo> classInfoMap = new ConcurrentHashMap<>();

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
                    JavaParserFacade facade = JavaParserFacade.get(combinedTypeSolver);
                    
                    int current = processedCount.incrementAndGet();
                    if (current % 100 == 0) {
                        System.out.println("已处理 " + current + "/" + totalFiles + " 个文件");
                    }
                    
                    analyzeDependencies(parser, facade, javaFile, classInfoMap, projectRootDir, projectPackagePrefix);
                } catch (Exception e) {
                    logError(javaFile, "分析文件失败", null, e, true);
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

        createParentDirectory(classInfoPath);
        writeClassInfoToFile(classInfoMap, classInfoPath);

        // 输出统计信息
        System.out.println("\n=== 统计信息 ===");
        System.out.println("总类数: " + classInfoMap.size());
        long withDependencies = classInfoMap.values().stream().filter(info -> !info.getDependencies().isEmpty()).count();
        long withExtendsAndImplements = classInfoMap.values().stream().filter(info -> !info.getExtendsAndImplements().isEmpty()).count();
        long withMethods = classInfoMap.values().stream().filter(info -> !info.getMethods().isEmpty()).count();
        System.out.println("有依赖关系的类数: " + withDependencies);
        System.out.println("有继承和实现关系的类数: " + withExtendsAndImplements);
        System.out.println("有方法信息的类数: " + withMethods);
        System.out.println("错误总数: " + errorMessages.size());
    }


    public static void createParentDirectory(String filePath) {
        // 1. 将文件路径转换为 Path 对象（Java NIO 推荐方式，更易用）
        Path filePathObj = Paths.get(filePath);

        // 2. 获取父文件夹路径（比如 D:/data/json）
        Path parentDir = filePathObj.getParent();

        // 3. 校验父路径是否为空（避免 filePath 是根路径/无父目录的情况）
        if (parentDir == null) {
            System.out.println("文件路径无父文件夹，无需创建");
            return;
        }

        // 4. 检查文件夹是否存在，不存在则创建（mkdirs() 会创建多级目录）
        if (!Files.exists(parentDir)) {
            try {
                // mkdirs()：创建多级目录（比如 data 不存在时，会同时创建 data 和 json）
                // 区别于 mkdir()：只能创建单级目录，父目录不存在则失败
                Files.createDirectories(parentDir);
                System.out.println("文件夹创建成功：" + parentDir);
            } catch (Exception e) {
                // 捕获异常（比如权限不足、路径非法等）
                System.err.println("创建文件夹失败：" + parentDir);
                e.printStackTrace();
            }
        } else {
            System.out.println("文件夹已存在：" + parentDir);
        }
    }

    /**
     * 分析单个文件的类依赖关系
     */
    private static void analyzeDependencies(JavaParser parser, JavaParserFacade facade,
                                                  File javaFile, Map<String, ClassInfo> classInfoMap,
                                                  File projectRootDir, String projectPackagePrefix) {
        // System.out.println("分析文件: " + javaFile.getAbsolutePath());
        try (FileInputStream in = new FileInputStream(javaFile)) {
            CompilationUnit cu = parser.parse(in).getResult().orElse(null);
            if (cu == null) {
                return;
            }

            // 获取当前文件的包名
            String packageName = cu.getPackageDeclaration()
                    .map(p -> p.getNameAsString())
                    .orElse("");
            
            // 提前判断：如果设置了包前缀，且包名不符合前缀，直接跳过整个文件
            // 注意：默认包（packageName 为空）无法提前判断，需要在后面检查类名
            if (projectPackagePrefix != null && !projectPackagePrefix.isEmpty() && !packageName.isEmpty()) {
                // 包名应该以 projectPackagePrefix 开头（加上 "."）或等于 projectPackagePrefix
                if (!packageName.startsWith(projectPackagePrefix + ".") && !packageName.equals(projectPackagePrefix)) {
                    return; // 包名不符合前缀，跳过整个文件
                }
            }
            
            // 获取所有顶级类型声明（类、接口、枚举、注解、Record）- 不包括内部类
            List<TypeDeclaration<?>> topLevelTypes = cu.getTypes();
            
            // 计算文件相对路径
            String fileRelativePath = getRelativePath(javaFile, projectRootDir);
            
            // 遍历所有顶级类型声明（类、接口、枚举、注解、Record）
            // 所有 TypeDeclaration 子类都可以统一处理，不需要区分
            for (TypeDeclaration<?> typeDecl : topLevelTypes) {
                try {
                    String typeName = getFullName(typeDecl, packageName);
                    
                    // 获取或创建 ClassInfo
                    ClassInfo classInfo = classInfoMap.computeIfAbsent(typeName, k -> {
                        ClassInfo info = new ClassInfo();
                        info.setQualifiedName(typeName);
                        info.setFilePath(fileRelativePath);
                        info.setTypeKind(getTypeKind(typeDecl));
                        return info;
                    });
                    
                    // 设置文件路径和类型（如果之前已存在但信息不完整）
                    if (classInfo.getFilePath() == null || classInfo.getFilePath().isEmpty()) {
                        classInfo.setFilePath(fileRelativePath);
                    }
                    if (classInfo.getTypeKind() == null || classInfo.getTypeKind().isEmpty()) {
                        classInfo.setTypeKind(getTypeKind(typeDecl));
                    }
                    
                    // 提取JavaDoc
                    if (typeDecl.getJavadoc().isPresent()) {
                        String javadocText = typeDecl.getJavadoc().get().toText();
                        if (!javadocText.isEmpty()) {
                            classInfo.setJavaDoc(javadocText);
                        }
                    }
                    
                    // 使用线程安全的集合来存储依赖
                    Set<String> dependencies = ConcurrentHashMap.newKeySet();
                    Set<String> extendsAndImplements = ConcurrentHashMap.newKeySet();
                    
                    // 专门提取继承和实现关系（包括内部类）
                    analyzeExtendsAndImplements(typeDecl, facade, extendsAndImplements, javaFile);
                    
                    // 提取顶级类型的方法签名（不包括内部类的方法）
                    analyzeMethods(typeDecl, facade, classInfo.getMethods(), javaFile);
                    
                    // 统一处理所有 TypeDeclaration：获取所有 Type 和 Expression
                    for (ClassOrInterfaceType type : typeDecl.findAll(ClassOrInterfaceType.class)) {
                        analyzeType(type, facade, dependencies, javaFile);
                    }
                    for (Expression expression : typeDecl.findAll(Expression.class)) {
                        analyzeExpression(expression, facade, dependencies, javaFile);
                    }
                    
                    // 将Set转换为List并过滤项目内依赖
                    classInfo.setDependencies(new ArrayList<>(filterProjectDependencies(dependencies, typeName)));
                    classInfo.setExtendsAndImplements(new ArrayList<>(filterProjectDependencies(extendsAndImplements, typeName)));

                } catch (Exception e) {
                    String typeName = getFullName(typeDecl, packageName);
                    logError(javaFile, "分析类型失败", typeName, e, false);
                }
            }

        } catch (Exception e) {
            logError(javaFile, "读取文件失败", null, e, true);
        }
    }

    private static void analyzeExtendsAndImplements(TypeDeclaration<?> typeDecl, JavaParserFacade facade, Set<String> extendsAndImplements, File javaFile) {
        // System.out.println("分析继承和实现关系: " + typeDecl.getNameAsString());
        
        try {
            for (ClassOrInterfaceDeclaration decl : typeDecl.findAll(ClassOrInterfaceDeclaration.class)) {
                // System.out.println("找到类定义: " + decl.getNameAsString());
                for (ClassOrInterfaceType type : decl.getExtendedTypes()) {
                    // System.out.println("找到继承类型: " + type.getNameAsString());
                    analyzeType(type, facade, extendsAndImplements, javaFile);
                }
                for (ClassOrInterfaceType type : decl.getImplementedTypes()) {
                    // System.out.println("找到实现类型: " + type.getNameAsString());
                    analyzeType(type, facade, extendsAndImplements, javaFile);
                }
            }
        } catch (Exception e) {
            logError(javaFile, "分析继承和实现关系失败", typeDecl.getNameAsString(), e, false);
        }
    }

    /**
     * 分析顶级类型的方法，提取完整的方法签名
     * 只提取顶级类型直接定义的方法，不包括内部类的方法
     */
    private static void analyzeMethods(TypeDeclaration<?> typeDecl, JavaParserFacade facade, List<String> methods, File javaFile) {
        try {
            // 只获取顶级类型直接定义的方法，不包括内部类的方法
            // 使用getMethods()而不是findAll()，因为findAll会递归查找所有方法包括内部类的
            List<MethodDeclaration> methodDecls = new ArrayList<>();
            
            if (typeDecl instanceof ClassOrInterfaceDeclaration) {
                ClassOrInterfaceDeclaration decl = (ClassOrInterfaceDeclaration) typeDecl;
                methodDecls.addAll(decl.getMethods());
            } else if (typeDecl instanceof EnumDeclaration) {
                EnumDeclaration enumDecl = (EnumDeclaration) typeDecl;
                methodDecls.addAll(enumDecl.getMethods());
            } else if (typeDecl instanceof AnnotationDeclaration) {
                AnnotationDeclaration annotationDecl = (AnnotationDeclaration) typeDecl;
                methodDecls.addAll(annotationDecl.getMethods());
            }
            
            // 处理所有找到的方法
            for (MethodDeclaration methodDecl : methodDecls) {
                try {
                    String methodSignature = methodDecl.getDeclarationAsString();
                    if (methodSignature != null && !methodSignature.isEmpty()) {
                        methods.add(methodSignature);
                    }
                } catch (Exception e) {
                    logError(javaFile, "分析方法签名失败", methodDecl.getNameAsString(), e, false);
                }
            }
        } catch (Exception e) {
            logError(javaFile, "分析方法失败", typeDecl.getNameAsString(), e, false);
        }
    }

    /**
     * 构建完整的方法签名
     * 格式: 返回类型 方法名(参数类型1, 参数类型2, ...)
     */
    private static String buildMethodSignature(MethodDeclaration methodDecl, JavaParserFacade facade) {
        try {
            StringBuilder signature = new StringBuilder();
            
            // 获取返回类型
            Type returnType = methodDecl.getType();
            String returnTypeStr = getTypeString(returnType, facade);
            signature.append(returnTypeStr).append(" ");
            
            // 获取方法名
            signature.append(methodDecl.getNameAsString());
            
            // 获取参数列表
            signature.append("(");
            NodeList<Parameter> parameters = methodDecl.getParameters();
            for (int i = 0; i < parameters.size(); i++) {
                if (i > 0) {
                    signature.append(", ");
                }
                Parameter param = parameters.get(i);
                Type paramType = param.getType();
                String paramTypeStr = getTypeString(paramType, facade);
                signature.append(paramTypeStr);
            }
            signature.append(")");
            
            return signature.toString();
        } catch (Exception e) {
            // 如果解析失败，尝试使用简单的方式构建签名
            try {
                StringBuilder signature = new StringBuilder();
                signature.append(methodDecl.getType().toString()).append(" ");
                signature.append(methodDecl.getNameAsString()).append("(");
                NodeList<Parameter> parameters = methodDecl.getParameters();
                for (int i = 0; i < parameters.size(); i++) {
                    if (i > 0) {
                        signature.append(", ");
                    }
                    signature.append(parameters.get(i).getType().toString());
                }
                signature.append(")");
                return signature.toString();
            } catch (Exception e2) {
                return null;
            }
        }
    }

    /**
     * 获取类型的字符串表示，尝试解析为完整限定名
     */
    private static String getTypeString(Type type, JavaParserFacade facade) {
        try {
            // 尝试解析类型获取完整限定名
            ResolvedType resolvedType = facade.convertToUsage(type);
            if (resolvedType.isReferenceType()) {
                return resolvedType.asReferenceType().getQualifiedName();
            } else if (resolvedType.isPrimitive()) {
                return resolvedType.asPrimitive().name().toLowerCase();
            } else if (resolvedType.isVoid()) {
                return "void";
            } else if (resolvedType.isArray()) {
                ResolvedArrayType arrayType = resolvedType.asArrayType();
                ResolvedType componentType = arrayType.getComponentType();
                String componentTypeStr = getTypeStringFromResolved(componentType);
                return componentTypeStr + "[]";
            } else {
                return type.toString();
            }
        } catch (Exception e) {
            // 解析失败，返回原始类型字符串
            return type.toString();
        }
    }

    /**
     * 从ResolvedType获取类型字符串
     */
    private static String getTypeStringFromResolved(ResolvedType resolvedType) {
        try {
            if (resolvedType.isReferenceType()) {
                return resolvedType.asReferenceType().getQualifiedName();
            } else if (resolvedType.isPrimitive()) {
                return resolvedType.asPrimitive().name().toLowerCase();
            } else if (resolvedType.isVoid()) {
                return "void";
            } else {
                return resolvedType.toString();
            }
        } catch (Exception e) {
            return resolvedType.toString();
        }
    }

    private static void analyzeType(ClassOrInterfaceType type, JavaParserFacade facade, Set<String> dependencies, File javaFile) {
        try {
            // System.out.println("分析类型: " + type.toString());
            if (type.getParentNode().isPresent() && type.getParentNode().get() instanceof ClassOrInterfaceType) {
                // 这是限定名的中间部分（包名），跳过，不解析
                return;
            }

            // 使用 facade 来解析类型，而不是直接调用 type.resolve()
            // 因为 facade 已经关联了 TypeSolver，可以正确解析类型
            ResolvedType resolvedType = facade.convertToUsage(type);
            // ResolvedType resolvedType = type.resolve();

            analyzeResolvedType(resolvedType, dependencies);
        } catch (Exception e) {
            // System.out.println("分析类型失败: " + type.toString() + " " + e.getMessage());
            if (type.getTypeArguments().isPresent()) {
                NodeList<Type> typeArgs = type.getTypeArguments().get();
            
                if (!typeArgs.isEmpty()) {
                    type.setTypeArguments(new NodeList<>());
                    
                    try {
                        analyzeType(type, facade, dependencies, javaFile);
                        return;
                    } catch (Exception e2) {
                        e = e2;
                    }
                }
            }

            // 内部类
            if (type.getScope().isPresent()) {
                try {
                    solveInnerClassChain(type, facade, dependencies);
                    return;
                } catch (Exception e1) {
                    // 无法解析，跳过
                }
            }

            // 检查是否是包名或变量名导致的错误（应该静默处理）
            String typeName = type.toString();
            
            // 其他类型解析错误，记录错误（可能是真正的配置问题）
            logError(javaFile, "分析类型失败", typeName, e, false);
        }
    }

    private static String solveInnerClassChain(ClassOrInterfaceType type, JavaParserFacade facade, Set<String> dependencies) {
        Optional<ClassOrInterfaceType> scope = type.getScope();
        if (scope.isPresent()) {
            String outerName = "";
            try {
                // 使用 facade 的方法来解析类型
                ResolvedType resolvedType = facade.convertToUsage(scope.get());
                ResolvedReferenceType refType = resolvedType.asReferenceType();
                outerName = refType.getQualifiedName();
                if (collectAllDependencies || determineTypeSolverSource(outerName).equals("JavaParserTypeSolver")) {
                    dependencies.add(outerName);
                } else {
                    return null;
                }
            } catch (Exception e) {
                outerName = solveInnerClassChain(scope.get(), facade, dependencies);
                if (outerName == null) {
                    return null;
                }
            }
            String thisQualifiedName = outerName + "." + type.getNameAsString();
            dependencies.add(thisQualifiedName);
            return  thisQualifiedName;
        } else {
            throw new RuntimeException("无法解析内部类链");
        }
    }
    
    private static void analyzeResolvedType(ResolvedType resolvedType, Set<String> dependencies) {
        try {
            ResolvedReferenceType refType = resolvedType.asReferenceType();
            String typeName = refType.getQualifiedName();
            if (collectAllDependencies || determineTypeSolverSource(typeName).equals("JavaParserTypeSolver")) {
                dependencies.add(typeName);
            }
            
            // 可选：判断类型来源（如果需要的话，可以取消注释下面的代码）
            // String typeSolverSource = determineTypeSolverSource(typeName);
            // System.out.println("类型 " + typeName + " 由 " + typeSolverSource + " 解析");
        } catch (Exception e) {
            // 无法解析，跳过
            throw e;
        }
    }
    
    /**
     * 判断类型是由哪个 TypeSolver 解析的
     * @param qualifiedName 类型的完整限定名
     * @return TypeSolver 类型字符串："ReflectionTypeSolver"、"JavaParserTypeSolver"、"JarTypeSolver" 或 "Unknown"
     */
    public static String determineTypeSolverSource(String qualifiedName) {
        if (qualifiedName == null || qualifiedName.isEmpty()) {
            return "Unknown";
        }
        
        // 对于 java.* 和 javax.* 包：Java 规范规定这些是保留包名，项目代码不能定义这些包下的类
        // 因此直接用 ReflectionTypeSolver 判断（最快且唯一可能）
        if (qualifiedName.startsWith("java.") || qualifiedName.startsWith("javax.")) {
            if (reflectionTypeSolver != null) {
                try {
                    SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                        reflectionTypeSolver.tryToSolveType(qualifiedName);
                    if (symbolRef.isSolved()) {
                        return "ReflectionTypeSolver";
                    }
                } catch (Exception e) {
                    // 解析失败，返回 Unknown
                }
            }
            // 如果 ReflectionTypeSolver 无法解析 java.* 或 javax.* 的类，返回 Unknown
            return "Unknown";
        }
        
        // 对于其他类：依次尝试每个 TypeSolver
        // 1. 先尝试 JavaParserTypeSolver（项目内类）- 这是我们最关心的
        for (JavaParserTypeSolver javaParserTypeSolver : javaParserTypeSolvers) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    javaParserTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "JavaParserTypeSolver";
                }
            } catch (Exception e) {
                // 解析失败，继续尝试下一个
            }
        }
        
        // 2. 再尝试 ReflectionTypeSolver（虽然不太可能，但为了完整性，比如某些非标准的 JDK 类）
        if (reflectionTypeSolver != null) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    reflectionTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "ReflectionTypeSolver";
                }
            } catch (Exception e) {
                // 解析失败，继续尝试下一个
            }
        }
        
        // 3. 最后尝试 JarTypeSolver（第三方依赖）
        for (JarTypeSolver jarTypeSolver : jarTypeSolvers) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    jarTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "JarTypeSolver";
                }
            } catch (Exception e) {
                // 解析失败，继续尝试下一个
            }
        }
        
        // 如果所有 TypeSolver 都无法解析，返回 Unknown
        return "Unknown";
    }

    private static void analyzeResolvedTypeDeclaration(ResolvedTypeDeclaration resolvedTypeDeclaration, Set<String> dependencies) {
        try {
            String typeName = resolvedTypeDeclaration.getQualifiedName();
            // 提取类型本身（如 ClassA）
            if (collectAllDependencies || determineTypeSolverSource(typeName).equals("JavaParserTypeSolver")) {
                dependencies.add(typeName);
            }
        } catch (Exception e) {
            // 无法解析，跳过
            throw e;
        }
    }

    private static void analyzeExpression(Expression expr, JavaParserFacade facade, Set<String> dependencies, File javaFile) {
        try {
            // 1. 注解表达式 - 需要记录注解类型
            if (expr instanceof AnnotationExpr annotationExpr) {
                try {
                    ResolvedAnnotationDeclaration resolved = annotationExpr.resolve();
                    String fqName = resolved.getQualifiedName(); // 注解类的全限定名
                    if (collectAllDependencies || determineTypeSolverSource(fqName).equals("JavaParserTypeSolver")) {
                        dependencies.add(fqName);
                    }
                } catch (Exception e) {
                    // 注解解析失败，记录错误（这是真正的错误，不是类型推断问题）
                    logError(javaFile, "分析注解失败", expr.toString(), e, false);
                }
                return;
            }
            
            // 2. 方法调用表达式 - 需要分析方法返回类型和方法声明所属类型
            // 这些信息无法通过 findAll(Type.class) 获取，因为返回类型可能不在类型声明中
            // scope 和参数会被 findAll(Expression.class) 找到，不需要递归处理
            if (expr instanceof MethodCallExpr methodCall) {
                try {
                    ResolvedMethodDeclaration methodDecl = methodCall.resolve();
                    // 分析方法返回类型（可能不在类型声明中）
                    analyzeResolvedType(methodDecl.getReturnType(), dependencies);
                    // 分析方法声明所属的类型（declaringType）
                    analyzeResolvedTypeDeclaration(methodDecl.declaringType(), dependencies);
                    // resolve() 成功，直接返回
                    return;
                } catch (Exception e) {
                    // resolve() 失败，不在这里处理，走到最后的默认 getType() 处理
                }
            }
            
            // 3. 字段访问表达式 - 需要分析字段类型和字段声明所属类型
            // 这些信息无法通过 findAll(Type.class) 获取
            // scope 会被 findAll(Expression.class) 找到，不需要递归处理
            if (expr instanceof FieldAccessExpr fieldAccess) {
                try {
                    // resolve() 返回 ResolvedValueDeclaration，需要转换为 ResolvedFieldDeclaration
                    ResolvedValueDeclaration valueDecl = fieldAccess.resolve();
                    if (valueDecl instanceof ResolvedFieldDeclaration fieldDecl) {
                        analyzeResolvedType(fieldDecl.getType(), dependencies);
                        // 分析字段声明所属的类型（declaringType）
                        analyzeResolvedTypeDeclaration(fieldDecl.declaringType(), dependencies);
                        // resolve() 成功，直接返回
                        return;
                    }
                } catch (Exception e) {
                    // resolve() 失败，不在这里处理，走到最后的默认 getType() 处理
                }
            }
            
            // 4. 方法引用表达式 - 需要分析引用的方法所属类型和函数式接口类型
            // 这些信息无法通过 findAll(Type.class) 获取
            // scope 会被 findAll(Expression.class) 找到，不需要递归处理
            if (expr instanceof MethodReferenceExpr methodRef) {
                // 分析方法声明（获取方法所属的类型）
                try {
                    ResolvedMethodDeclaration methodDecl = methodRef.resolve();
                    // 分析方法声明所属的类型（declaringType）
                    analyzeResolvedTypeDeclaration(methodDecl.declaringType(), dependencies);
                    return;
                } catch (Exception e) {
                    // resolve() 失败，不在这里处理，走到最后的默认 getType() 处理
                }
            }
            
            // 对于其他表达式，统一使用 getType() 尝试解析类型
            // 包括：
            // - 方法调用表达式（resolve() 失败的情况）
            // - 字段访问表达式（resolve() 失败的情况）
            // - Lambda 表达式（函数式接口类型）
            // - 对象创建表达式
            // - 三元运算符、二元表达式等（虽然会失败，但统一处理）
            // 这些表达式的子表达式/子类型都已经被 findAll 处理了，这里只需要尝试解析表达式本身的类型
            // 如果解析失败，静默跳过（因为这些表达式可能不包含类型依赖信息）
            try {
                ResolvedType resolvedType = facade.getType(expr);
                if (resolvedType != null) {
                    analyzeResolvedType(resolvedType, dependencies);
                }
            } catch (Exception typeResolveException) {
                // 对于无法解析类型的表达式，静默跳过
            }
        } catch (Exception e) {
            // 检查是否是类型解析相关的错误（应该静默处理）
            String exprStr = expr.toString();
            
            // 其他异常（如空指针、配置问题等），记录错误
            logError(javaFile, "分析表达式失败", exprStr, e, false);
        }
    }

    /**
     * 获取类型声明的完整名称（类、接口、枚举、注解）
     */
    private static String getFullName(TypeDeclaration<?> n, String packageName) {
        String typeName = n.getNameAsString();
        if (!packageName.isEmpty()) {
            return packageName + "." + typeName;
        }
        return typeName;
    }
    
    /**
     * 获取类型种类（class、interface、enum、annotation）
     */
    private static String getTypeKind(TypeDeclaration<?> typeDecl) {
        if (typeDecl instanceof ClassOrInterfaceDeclaration) {
            ClassOrInterfaceDeclaration decl = (ClassOrInterfaceDeclaration) typeDecl;
            return decl.isInterface() ? "interface" : "class";
        } else if (typeDecl instanceof EnumDeclaration) {
            return "enum";
        } else if (typeDecl instanceof AnnotationDeclaration) {
            return "annotation";
        } else {
            // 可能是 Record 或其他类型
            return "class";
        }
    }
    
    /**
     * 获取文件相对于项目根目录的相对路径
     */
    private static String getRelativePath(File file, File projectRootDir) {
        try {
            Path filePath = file.toPath().toAbsolutePath().normalize();
            Path rootPath = projectRootDir.toPath().toAbsolutePath().normalize();
            Path relativePath = rootPath.relativize(filePath);
            // 将路径分隔符统一为 "/"
            return relativePath.toString().replace(File.separator, "/");
        } catch (Exception e) {
            // 如果计算相对路径失败，返回绝对路径
            return file.getAbsolutePath();
        }
    }
    
    /**
     * 过滤依赖，只保留项目内的类（符合项目包前缀的类）
     */
    private static Set<String> filterProjectDependencies(Set<String> dependencies, String sourceClass) {
        if (projectPackagePrefix == null || projectPackagePrefix.isEmpty()) {
            // 如果没有设置包前缀，返回所有依赖
            return dependencies;
        }
        
        Set<String> filtered = new HashSet<>();
        for (String dependency : dependencies) {
            // 只保留以项目包前缀开头的依赖
            if ((dependency.startsWith(projectPackagePrefix + ".") || dependency.equals(projectPackagePrefix)) && !dependency.equals(sourceClass)) {
                filtered.add(dependency);
            }
        }
        return filtered;
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


    public static void loadStaticDependencies(CombinedTypeSolver combinedTypeSolver, File projectRootDir) {

        // 递归查找所有 microweaver-static-dependency 文件夹
        List<File> dependencyDirs = findAllDependencyDirs(projectRootDir);
        
        if (dependencyDirs.isEmpty()) {
            System.out.println("未找到任何 microweaver-static-dependency 目录，正在执行 Maven 命令下载依赖...");
            
            // 查找所有包含 pom.xml 的目录
            System.out.println("正在查找所有包含 pom.xml 的目录...");
            List<File> pomDirs = findAllPomDirs(projectRootDir);
            System.out.println("找到 " + pomDirs.size() + " 个包含 pom.xml 的目录");
            
            // 在每个包含 pom.xml 的目录执行 Maven 命令
            for (File pomDir : pomDirs) {
                System.out.println("在目录执行 Maven 命令: " + pomDir.getAbsolutePath());
                executeMavenCopyDependencies(pomDir);
            }
            
            // 重新查找依赖目录
            dependencyDirs = findAllDependencyDirs(projectRootDir);
        }
        
        // 从所有依赖目录中收集 JAR 文件，并去重（通过文件名）
        // 使用 Map 来去重：key 是文件名，value 是 File 对象
        // 这样可以避免加载重复的 JAR 文件（不同模块可能有相同版本的依赖）
        // 注意：如果不同模块使用了不同版本的同一依赖，只保留第一个遇到的版本
        Map<String, File> uniqueJarFiles = new HashMap<>();
        int totalJarCount = 0;
        
        System.out.println("找到 " + dependencyDirs.size() + " 个 microweaver-static-dependency 目录:");
        for (File dependencyDir : dependencyDirs) {
            File[] jars = dependencyDir.listFiles((dir, name) -> name.endsWith(".jar"));
            int jarCount = jars != null ? jars.length : 0;
            totalJarCount += jarCount;
            System.out.println("  - " + dependencyDir.getAbsolutePath() + " (" + jarCount + " 个 JAR 文件)");
            if (jars != null) {
                for (File jar : jars) {
                    // 使用文件名作为key进行去重
                    // 相同版本的依赖会有相同的文件名（如 commons-lang3-3.12.0.jar）
                    String jarName = jar.getName();
                    if (!uniqueJarFiles.containsKey(jarName)) {
                        uniqueJarFiles.put(jarName, jar);
                    }
                }
            }
        }
        
        System.out.println("去重前总 JAR 文件数: " + totalJarCount);
        System.out.println("去重后唯一 JAR 文件数: " + uniqueJarFiles.size());
        if (totalJarCount > uniqueJarFiles.size()) {
            System.out.println("已去除 " + (totalJarCount - uniqueJarFiles.size()) + " 个重复的 JAR 文件");
        }
        
        // 实际加载 JAR 文件到 CombinedTypeSolver，并收集成功加载的依赖信息
        int loadedCount = 0;
        int failedCount = 0;
        List<String> loadedDependencies = new ArrayList<>();
        
        for (File jarFile : uniqueJarFiles.values()) {
            try {
                JarTypeSolver jarTypeSolver = new JarTypeSolver(jarFile);
                jarTypeSolvers.add(jarTypeSolver);
                combinedTypeSolver.add(jarTypeSolver);
                loadedCount++;
                // 收集成功加载的依赖名称
                loadedDependencies.add(jarFile.getName());
            } catch (Exception e) {
                failedCount++;
                logError(null, "加载 JAR 文件失败", jarFile.getName(), e, false);
            }
        }
        
        System.out.println("成功加载 " + loadedCount + " 个 JAR 文件到类型解析器");
        if (failedCount > 0) {
            System.out.println("加载失败 " + failedCount + " 个 JAR 文件");
        }
        
//         // 保存成功加载的依赖名称到 JSON 文件
//         try {
//             Gson gson = new GsonBuilder().setPrettyPrinting().create();
//             String json = gson.toJson(loadedDependencies);
//
//             try (FileWriter writer = new FileWriter("output/static-dependencies.json")) {
//                 writer.write(json);
//                 System.out.println("静态依赖信息已保存到 static-dependencies.json（共 " + loadedDependencies.size() + " 个依赖）");
//             }
//         } catch (IOException e) {
//             logError(null, "保存静态依赖信息失败", null, e, false);
//         }
    }


    /**
     * 执行 Maven dependency:copy-dependencies 命令
     * @param dir 执行命令的目录
     */
    private static void executeMavenCopyDependencies(File dir) {
        try {
            File microweaverStaticDir = new File(dir, "microweaver-static-dependency");
            if (microweaverStaticDir.exists() && microweaverStaticDir.isDirectory()) {
                // 如果已存在 microweaver-static-dependency 目录，则跳过执行 Maven 命令
                return;
            }
            List<String> cmd = System.getProperty("os.name").toLowerCase().contains("win")
                ? List.of("cmd", "/c", "mvn", "dependency:copy-dependencies", 
                          "-DoutputDirectory=microweaver-static-dependency",
                          "-DincludeScope=compile", 
                          "-DexcludeTransitive=false",
                          "-q")  // -q 参数减少输出
                : List.of("mvn", "dependency:copy-dependencies", 
                          "-DoutputDirectory=microweaver-static-dependency",
                          "-DincludeScope=compile",
                          "-DexcludeTransitive=false",
                          "-q");
            Process process = new ProcessBuilder(cmd)
                .directory(dir)
                .inheritIO()  // 让 Maven 输出直接显示在控制台
                .start();
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.out.println("警告: 在目录 " + dir.getAbsolutePath() + " 执行 Maven 命令返回非零退出码: " + exitCode);
            }
        } catch (Exception e) {
            logError(null, "执行 Maven 命令失败", dir.getAbsolutePath(), e, false);
        }
    }
    
    /**
     * 递归查找所有包含 pom.xml 的目录
     * @param rootDir 根目录
     * @return 包含 pom.xml 的目录列表
     */
    private static List<File> findAllPomDirs(File rootDir) {
        List<File> pomDirs = new ArrayList<>();
        findPomDirsRecursively(rootDir, pomDirs);
        return pomDirs;
    }
    
    /**
     * 递归查找包含 pom.xml 的目录的辅助方法
     * @param dir 当前目录
     * @param pomDirs 结果列表
     */
    private static void findPomDirsRecursively(File dir, List<File> pomDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // 检查当前目录是否包含 pom.xml
        File pomFile = new File(dir, "pom.xml");
        if (pomFile.exists() && pomFile.isFile()) {
            pomDirs.add(dir);
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
                        findPomDirsRecursively(file, pomDirs);
                    }
                }
            }
        }
    }
    
    /**
     * 递归查找项目根目录下所有的 microweaver-static-dependency 目录
     */
    private static List<File> findAllDependencyDirs(File rootDir) {
        List<File> dependencyDirs = new ArrayList<>();
        findDependencyDirsRecursively(rootDir, dependencyDirs);
        return dependencyDirs;
    }

    /**
     * 递归查找 microweaver-static-dependency 目录的辅助方法
     */
    private static void findDependencyDirsRecursively(File dir, List<File> dependencyDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // 检查当前目录是否是 microweaver-static-dependency
        if (dir.getName().equals("microweaver-static-dependency")) {
            dependencyDirs.add(dir);
            return; // 找到后不再深入，避免重复
        }
        
        // 递归查找子目录
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    findDependencyDirsRecursively(file, dependencyDirs);
                }
            }
        }
    }

    
    /**
     * 统一的错误日志记录函数
     * @param javaFile 出错的Java文件（可为null）
     * @param errorType 错误类型描述（如"分析文件失败"、"分析类型失败"等）
     * @param details 错误详情（如类型名、表达式等）
     * @param e 异常对象（可为null）
     * @param includeStackTrace 是否包含堆栈跟踪
     */
    private static void logError(File javaFile, String errorType, String details, Exception e, boolean includeStackTrace) {
        if (e != null && (e instanceof IllegalStateException || e instanceof UnsupportedOperationException)) {
            return;
        }
        StringBuilder errorMsg = new StringBuilder();
        
        // 添加文件信息
        if (javaFile != null) {
            errorMsg.append("文件: ").append(javaFile.getAbsolutePath());
            errorMsg.append(" | ");
        }
        
        // 添加错误类型
        errorMsg.append(errorType);
        
        // 添加详情
        if (details != null && !details.isEmpty()) {
            errorMsg.append(": ").append(details);
        }
        
        // 添加异常信息
        if (e != null) {
            // 添加异常类型和消息
            errorMsg.append(" | 异常类型: ").append(e.getClass().getSimpleName());
            errorMsg.append(" | 错误: ").append(e.getMessage());
        }
        
        String errorMessage = errorMsg.toString();
        
        // 输出到控制台
        System.err.println(errorMessage);
        
        // 添加到错误收集器
        errorMessages.offer(errorMessage);
        
        // 如果需要，添加堆栈跟踪
        if (includeStackTrace && e != null) {
            StringBuilder stackTrace = new StringBuilder();
            stackTrace.append(errorMessage).append("\n");
            for (StackTraceElement element : e.getStackTrace()) {
                stackTrace.append("    at ").append(element.toString()).append("\n");
            }
            errorMessages.offer(stackTrace.toString());
        }
    }

    private static void deleteDependencyDirsRecursively(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // 检查当前目录是否是 microweaver-static-dependency
        if (dir.getName().equals("microweaver-static-dependency")) {
            // Java 7+ 可以用 java.nio.file.Files.delete(Path) 递归删除整个目录
            try {
                java.nio.file.Files.walk(dir.toPath())
                    .sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            java.nio.file.Files.delete(path);
                        } catch (Exception e) {
                            // 忽略删除出错
                        }
                    });
            } catch (Exception e) {
                // 忽略异常
            }
            return; // 找到后不再深入，避免重复
        }
        
        // 递归查找子目录
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDependencyDirsRecursively(file);
                }
            }
        }
    }

    /**
     * 统一输出类信息到一个JSON文件
     */
    private static void writeClassInfoToFile(Map<String, ClassInfo> classInfoMap, String fileName) {
        // 保存为 JSON，禁用 HTML 转义以便直接显示 < 和 >
        Gson gson = new GsonBuilder()
                .setPrettyPrinting()
                .disableHtmlEscaping()
                .create();
        String json = gson.toJson(classInfoMap);

        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(json);
            System.out.println(fileName + "已保存（共 " + classInfoMap.size() + " 个类）");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}